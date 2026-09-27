"""Audit local H200 run evidence and export small, reproducible result tables.

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
    parser.add_argument('--original-run', type=Path, required=True)
    parser.add_argument('--resumed-run', type=Path, required=True)
    parser.add_argument('--ray-log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    original_log = args.original_run/'train-train.log'
    initial = parse_metrics(original_log)
    resumed = parse_metrics(args.ray_log)
    assert set(resumed) == set(range(51, 1740)), 'Missing/extra resumed updates'
    rows = {i: initial[i] for i in range(1,51)} | resumed
    for step, r in rows.items():
        assert r['rollout/generated_sequences'] == 128 and r['tip/global_rollouts'] == 128
        assert r['tip/global_normalization'] == 1
        assert math.isfinite(r['opd/loss']) and math.isfinite(r['opd/grad_norm'])
        assert 0.48 < r['tip/selected_fraction'] <= 0.5
    out = next((args.resumed_run/'train-output').glob('Qwen3-4B-*'))
    old_out = next((args.original_run/'train-output').glob('Qwen3-4B-*'))
    assert (out/'latest_checkpointed_iteration.txt').read_text().strip() == '1739'
    final = out/'global_step_1739'
    inventory = []
    for f in sorted(final.rglob('*')):
        if f.is_file():
            inventory.append({'file': str(f.relative_to(final)), 'bytes': f.stat().st_size})
    for rank in range(4):
        for kind in ('model','optim','extra_state'):
            f = final/'actor'/f'{kind}_world_size_4_rank_{rank}.pt'
            assert f.stat().st_size > 0
    assert (final/'data.pt').stat().st_size > 0
    eval_steps = list(range(50,1701,50)) + [1739]
    eval_rows, evidence = [], []
    benches = [('math',500),('aime24',30),('aime25',30)]
    for step in eval_steps:
        f = (old_out if step==50 else out)/f'{step}.jsonl'
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
    gpu_log = (args.resumed_run/'gpu-resume.log').read_text(errors='replace')
    assert 'resume training PASS' in gpu_log
    assert 'finished with exit=5' in gpu_log
    summary = {
        'training_completed':True,'final_step':1739,'rollouts_per_step':128,
        'total_training_rollouts':1739*128,'full_launcher_success':False,
        'gpu_wrapper_exit_code':5,'exit_reason':'Node-wide /root and /tmp growth exceeded 64 MiB cleanup guard after successful training',
        'overlay_growth_bytes':{'root':77902806,'tmp':109314414},
        'keepalive_restored_in_log':True,'original_run':str(args.original_run),'resumed_run':str(args.resumed_run),
        'authoritative_resumed_log':str(args.ray_log),'final_checkpoint':str(final),
        'final_evaluation':final_eval,'per_benchmark_best':best,
        'best_macro_checkpoint':max(eval_rows,key=lambda r:r['macro_mean']),
        'evaluation_count':len(eval_rows),'outputs_per_evaluation':8960,
        'normal_step_mean_s':statistics.mean(r['timing/step_s'] for r in normal),
        'normal_step_median_s':statistics.median(r['timing/step_s'] for r in normal),
        'normal_generate_mean_s':statistics.mean(r['timing/generate_s'] for r in normal),
        'normal_update_mean_s':statistics.mean(r['timing/train_s'] for r in normal),
        'summed_retained_step_hours':sum(r['timing/step_s'] for r in rows.values())/3600,
        'resumed_step_hours':sum(r['timing/step_s'] for r in resumed.values())/3600,
        'peak_reported_torch_allocated_gib':max(r['perf/max_memory_allocated_gb'] for r in rows.values()),
        'final_training_metrics':{k:v for k,v in rows[1739].items() if not k.startswith('val-')},
        'limitations':['Original non-GRPO teacher differs from paper','No step-0 evaluation or teacher evaluation','One run only; best checkpoint selection uses the reported evaluation benchmarks','Torch allocated-memory metric excludes SGLang and total device reservation','Checkpoint weights checked for existence/size; no final inference reload performed']}
    target=args.output_dir;target.mkdir(parents=True,exist_ok=True)
    for name, obj in [('summary.json',summary),('checkpoint_inventory.json',inventory),('evaluation_evidence.json',evidence)]:
        (target/name).write_text(json.dumps(obj,indent=2)+'\n')
    for name, data in [('eval_curve.csv',eval_rows),('training_metrics.csv',[{'global_step':i,**{k:v for k,v in r.items() if not k.startswith('val-')}} for i,r in rows.items()])]:
        keys=list(dict.fromkeys(k for r in data for k in r))
        with (target/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");writer.writeheader();writer.writerows(data)
    print(json.dumps({k:summary[k] for k in ['training_completed','full_launcher_success','final_evaluation','best_macro_checkpoint','normal_step_mean_s','summed_retained_step_hours']},indent=2))


if __name__ == '__main__':
    main()
