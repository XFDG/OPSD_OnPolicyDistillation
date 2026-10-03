"""CPU audit of a completed optimized run; export aggregates, never model text.

Extends the original full B200 audit without changing training or GPU state.
Run on the paired GPU Pod with CUDA_VISIBLE_DEVICES="" and persistent output.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import summarize_b200_run as base


def timestamped_events(text):
    events = []
    for line in text.splitlines():
        match = re.search(r'\[opsd\] (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)', line)
        if match and any(word in line for word in (
            'mode=', 'starting smoke', 'starting train', 'training PASS',
            'finished with exit=', 'launcher exit=', 'overlay growth:',
            'GPU keep-alive active',
        )):
            events.append({'utc': match[1], 'line': line.strip()})
    return events


def elapsed(events, start_word, end_word):
    start = next(event['utc'] for event in events if start_word in event['line'])
    end = next(event['utc'] for event in reversed(events) if end_word in event['line'])
    hours = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()/3600
    return {'start_utc': start, 'end_utc': end, 'hours': hours}


def main():
    # Reuse the original 1739-update, all-checkpoint, all-JSONL and CPU state audit.
    base.main()
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    target = Path(args['--output-dir'])
    run = Path(args['--run'])
    summary = json.loads((target/'summary.json').read_text())
    profile = json.loads((run/'profile.json').read_text())
    with (target/'training_metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1739
    for row in rows:
        assert float(row['opt/teacher_cache_cuda']) == 1
        assert float(row['opt/resident_teacher']) == 1
    assert profile['profile']['experimental_overlap'] is False
    assert profile['profile']['profile'] is False
    repo = Path(__file__).resolve().parents[1]
    mismatches = [name for name, expected in profile['source_sha256'].items()
                  if not (repo/name).is_file() or base.digest(repo/name) != expected]
    assert not mismatches, mismatches
    events = timestamped_events((run/'driver.log').read_text(errors='replace'))
    train_wall = elapsed(events, 'starting train', 'train training PASS')
    full_wall = elapsed(events, 'mode=all;', 'launcher exit=0')
    out = Path(summary['final_checkpoint']).parent
    checkpoint_steps = sorted(int(p.name.rsplit('_',1)[1]) for p in out.glob('global_step_*'))
    expected_steps = list(range(50,1701,50)) + [1739]
    assert checkpoint_steps == expected_steps
    assert '=== OPD training completed ===' in (run/'train-train.log').read_text(errors='replace')

    # Per-question aggregates allow later pairing without publishing prompt/answer text.
    with (out/'1739.jsonl').open() as stream:
        values = [json.loads(line) for line in stream]
    question_rows, offset = [], 0
    for bench, count in [('math',500),('aime24',30),('aime25',30)]:
        part = values[offset:offset+count*16]
        for index in range(count):
            group = part[index*16:(index+1)*16]
            assert all(item['acc'] in (0,1) for item in group)
            question_rows.append({'benchmark':bench,'question_index':index,
                                  'input_sha256':hashlib.sha256(group[0]['input'].encode()).hexdigest(),
                                  'samples':16,'correct':sum(item['acc'] for item in group),
                                  'mean_accuracy':sum(item['acc'] for item in group)/16})
        offset += count*16
    with (target/'final_question_scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(question_rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(question_rows)

    # Only small extra-state metadata are loaded, never weight/optimizer shards.
    import torch
    state=json.loads((target/'checkpoint_state_verification.json').read_text())
    for rank, entry in enumerate(state['ranks']):
        extra=torch.load(Path(summary['final_checkpoint'])/'actor'/f'extra_state_world_size_8_rank_{rank}.pt',
                         map_location='cpu',weights_only=False)
        entry['rng_keys']=sorted(extra['rng']) if isinstance(extra['rng'],dict) else None
    (target/'checkpoint_state_verification.json').write_text(json.dumps(state,indent=2)+'\n')

    keepalive=[]
    training=[]
    processes={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            comm=(p/'comm').read_text().strip()
            status=(p/'status').read_text()
            parent=int(re.search(r'^PPid:\s+(\d+)',status,re.M)[1])
        except (OSError,TypeError):continue
        processes[int(p.name)]={'pid':int(p.name),'comm':comm,'ppid':parent}
        if any(arg.endswith('/gpu-workspace/keep_alive/main.py') for arg in argv):keepalive.append(int(p.name))
        if comm.startswith(('sglang::','ray::WorkerDict','ray::OPDTask')):training.append(processes[int(p.name)])
    process_state={'collected_utc':datetime.now(timezone.utc).isoformat(),
                   'keepalive_parents':keepalive,
                   'keepalive_children':[item for item in processes.values() if item['ppid'] in keepalive],
                   'visible_training_workers':training}
    (target/'process_state.json').write_text(json.dumps(process_state,indent=2)+'\n')
    summary.update(
        collected_utc=datetime.now(timezone.utc).isoformat(),
        optimized_profile=profile['profile'],source_files_verified=len(profile['source_sha256']),
        source_mismatches=mismatches,all_updates_used_cuda_cache_and_resident_teacher=True,
        training_shell_wall=train_wall,full_launcher_wall=full_wall,
        total_training_response_tokens=sum(int(float(row['tip/global_response_tokens'])) for row in rows),
        mean_training_response_tokens=sum(float(row['tip/global_response_tokens']) for row in rows)/len(rows),
        benchmark_question_count=560,normal_step_count=1704,
        normal_step_filter='exclude steps 50,100,...,1700,1739; include initialization step1',
        quality_scope='Final step1739 full-benchmark mean@16; single run, not a statistical non-inferiority test',
        current_keepalive_parent_count=len(keepalive),visible_training_worker_count=len(training),
    )
    summary['limitations'].extend([
        'GPU2 NVLink was abnormal in prior read-only snapshots; no repair or final port-state recheck in this audit',
        'Reported microcode equality does not establish hardware health or firmware fix applicability',
        'Only extra-state and dataloader CPU reload; model/optimizer weights checked by inventory, not final inference reload',
    ])
    (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (target/'run_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (target/'completion_events.json').write_text(json.dumps(events,indent=2)+'\n')
    manifest=json.loads((target/'source_manifest.json').read_text())
    for path in [run/'profile.json',Path(__file__),repo/'scripts/summarize_b200_run.py']:
        manifest.append({'file':str(path),'sha256':base.digest(path)})
    (target/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({key:summary[key] for key in [
        'training_completed','final_evaluation','best_macro_checkpoint','normal_step_mean_s',
        'normal_update_mean_s','summed_retained_step_hours','full_launcher_wall',
        'source_files_verified','current_keepalive_parent_count','visible_training_worker_count',
    ]},indent=2))


if __name__ == '__main__':
    main()
