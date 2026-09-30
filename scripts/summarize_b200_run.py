"""Audit local B200 run evidence and export small, reproducible result tables.

No model execution, GPU allocation, checkpoint mutation, or raw sample export.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def parse_metrics(path):
    rows = {}
    for line in path.read_text(errors='replace').splitlines():
        found = re.search(r'step:(\d+) - ', line)
        if found:
            rows[int(found[1])] = {k: float(v) for k, v in re.findall(r'([a-zA-Z0-9_/@-]+):([-+0-9.eE]+)', line)}
    return rows


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ray-log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = parse_metrics(args.ray_log)
    assert set(rows) == set(range(1,1740)), 'Missing/extra updates'
    for step, r in rows.items():
        assert r['rollout/generated_sequences'] == 128 and r['tip/global_rollouts'] == 128
        assert r['tip/global_normalization'] == 1
        assert math.isfinite(r['opd/loss']) and math.isfinite(r['opd/grad_norm'])
        assert 0.48 < r['tip/selected_fraction'] <= 0.5
    out = next((args.run/'train-output').glob('Qwen3-4B-*'))
    assert (out/'latest_checkpointed_iteration.txt').read_text().strip() == '1739'
    final = out/'global_step_1739'
    inventory = []
    for f in sorted(final.rglob('*')):
        if f.is_file():
            inventory.append({'file': str(f.relative_to(final)), 'bytes': f.stat().st_size})
    for rank in range(8):
        for kind in ('model','optim','extra_state'):
            f = final/'actor'/f'{kind}_world_size_8_rank_{rank}.pt'
            assert f.stat().st_size > 0
    assert (final/'data.pt').stat().st_size > 0
    eval_steps = list(range(50,1701,50)) + [1739]
    for checkpoint_step in eval_steps:
        for rank in range(8):
            for kind in ('model','optim','extra_state'):
                assert (out/f'global_step_{checkpoint_step}'/'actor'/f'{kind}_world_size_8_rank_{rank}.pt').stat().st_size > 0
    eval_rows, evidence = [], []
    benches = [('math',500),('aime24',30),('aime25',30)]
    for step in eval_steps:
        f = out/f'{step}.jsonl'
        # Iterate actual JSONL records; model text can contain U+2028/U+0085.
        # str.splitlines() would incorrectly split those inside JSON strings.
        with f.open() as handle:
            values = [json.loads(line) for line in handle]
        assert len(values) == 8960, (step,len(values))
        row, offset = {'step': step}, 0
        for bench, questions in benches:
            part = values[offset:offset+questions*16]
            for i in range(0,len(part),16):
                assert len({v['input'] for v in part[i:i+16]}) == 1
            mean = statistics.mean(v['acc'] for v in part)
            assert math.isclose(mean, rows[step][f'val-core/{bench}/acc/mean@16'], abs_tol=1e-12)
            row[bench+'_mean_at_16'] = mean
            offset += questions*16
        row['macro_mean'] = statistics.mean(row[b+'_mean_at_16'] for b,_ in benches)
        eval_rows.append(row)
        evidence.append({'file':str(f),'bytes':f.stat().st_size,'sha256':digest(f)})
    normal = [r for step,r in rows.items() if step not in eval_steps]
    final_eval = eval_rows[-1]
    best = {b: max(eval_rows,key=lambda r:r[b+'_mean_at_16']) for b,_ in benches}
    gpu_log = (args.run/'gpu-train.log').read_text(errors='replace')
    driver = (args.run/'driver.log').read_text(errors='replace')
    assert 'train training PASS' in gpu_log and 'GPU train finished with exit=0' in gpu_log
    assert 'launcher exit=0' in driver
    assert 'GPU keep-alive active' in driver
    from datetime import datetime
    timestamps = re.findall(r'\[opsd\] (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)', driver)
    launch_wall_hours = (datetime.fromisoformat(timestamps[-1])-datetime.fromisoformat(timestamps[0])).total_seconds()/3600
    import torch
    state_report = {'ranks': []}
    for rank in range(8):
        extra = torch.load(final/'actor'/f'extra_state_world_size_8_rank_{rank}.pt', map_location='cpu', weights_only=False)
        scheduler = extra['lr_scheduler']
        assert scheduler['last_epoch'] == 1739 and all(x == 0 for x in scheduler['_last_lr'])
        assert 'rng' in extra
        state_report['ranks'].append({'rank':rank,'scheduler_step':scheduler['last_epoch'],'next_lr':scheduler['_last_lr'],'rng_present':True})
    data = torch.load(final/'data.pt', map_location='cpu', weights_only=False)
    cursor = data['_snapshot']['_snapshot_step'] + data['_steps_since_snapshot']
    yielded = data['_snapshot']['_main_snapshot']['_sampler_iter_state']['samples_yielded']
    assert cursor == 1739 and yielded == 13912
    state_report.update(data_cursor=cursor,prompts_yielded=yielded,passed=True)
    growth = re.findall(r'overlay growth: /root=(-?\d+) bytes; /tmp=(-?\d+) bytes',gpu_log)[-1]
    summary = {
        'training_completed':True,'final_step':1739,'rollouts_per_step':128,
        'total_training_rollouts':1739*128,'full_launcher_success':True,
        'gpu_wrapper_exit_code':0,'exit_reason':'Training, evaluation and cleanup completed successfully',
        'overlay_growth_bytes':dict(zip(['root','tmp'],map(int,growth))),
        'keepalive_restored_in_log':True,'run':str(args.run),
        'authoritative_log':str(args.ray_log),'final_checkpoint':str(final),
        'final_evaluation':final_eval,'per_benchmark_best':best,
        'best_macro_checkpoint':max(eval_rows,key=lambda r:r['macro_mean']),
        'checkpoint_count':len(eval_steps),'launcher_wall_hours_including_smoke':launch_wall_hours,'evaluation_count':len(eval_rows),'outputs_per_evaluation':8960,
        'normal_step_mean_s':statistics.mean(r['timing/step_s'] for r in normal),
        'normal_step_median_s':statistics.median(r['timing/step_s'] for r in normal),
        'normal_generate_mean_s':statistics.mean(r['timing/generate_s'] for r in normal),
        'normal_update_mean_s':statistics.mean(r['timing/train_s'] for r in normal),
        'summed_retained_step_hours':sum(r['timing/step_s'] for r in rows.values())/3600,
        'peak_reported_torch_allocated_gib':max(r['perf/max_memory_allocated_gb'] for r in rows.values()),
        'final_training_metrics':{k:v for k,v in rows[1739].items() if not k.startswith('val-')},
        'limitations':['Original non-GRPO teacher differs from paper','No step-0 evaluation or teacher evaluation','One run only; best checkpoint selection uses the reported evaluation benchmarks','Torch allocated-memory metric excludes SGLang and total device reservation','Checkpoint weights checked for existence/size; no final inference reload performed']}
    target=args.output_dir;target.mkdir(parents=True,exist_ok=True)
    for name, obj in [('summary.json',summary),('checkpoint_inventory.json',inventory),('evaluation_evidence.json',evidence),('checkpoint_state_verification.json',state_report),('source_manifest.json',[{'file':str(p),'sha256':digest(p)} for p in [args.ray_log,args.run/'driver.log',args.run/'gpu-train.log',*sorted((Path(__file__).resolve().parents[1]/'src/opd').glob('*.py'))]])]:
        (target/name).write_text(json.dumps(obj,indent=2)+'\n')
    for name, data in [('eval_curve.csv',eval_rows),('training_metrics.csv',[{'global_step':i,**{k:v for k,v in r.items() if not k.startswith('val-')}} for i,r in rows.items()])]:
        keys=list(dict.fromkeys(k for r in data for k in r))
        with (target/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");writer.writeheader();writer.writerows(data)
    (target/'completion-evidence.log').write_text('\n'.join(line for line in driver.splitlines() if '[opsd]' in line and any(x in line for x in ['mode=','[4/4]','training PASS','finished with exit=','overlay growth:','launcher exit=','GPU keep-alive active']))+'\n')
    print(json.dumps({k:summary[k] for k in ['training_completed','full_launcher_success','final_evaluation','best_macro_checkpoint','normal_step_mean_s','summed_retained_step_hours']},indent=2))


if __name__ == '__main__':
    main()
