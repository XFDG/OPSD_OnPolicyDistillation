"""Verify checkpoint restoration, schedule continuity and two real updates."""
import json
import math
from pathlib import Path
import re
import sys
import torch
run, source = map(Path, sys.argv[1:])
start = int(source.name.rsplit('_', 1)[1])
log = re.sub(r'\x1b\[[0-9;]*m', '', (run/'resume-smoke-train.log').read_text(errors='replace'))
assert f'Setting global step to {start}' in log
assert 'Total training steps: 1739' in log
live = run / 'ray-live.log'
if live.exists(): log += '\n' + live.read_text(errors='replace')
steps = {}
for line in log.splitlines():
    m = re.search(r'step:(\d+) - ', line)
    if m:
        steps[int(m[1])] = {k: float(v) for k, v in re.findall(r'([a-zA-Z0-9_/@-]+):([-+0-9.eE]+)',line)}
assert set(steps) == {start+1,start+2}, steps.keys()
initial = torch.load(source/'actor/extra_state_world_size_8_rank_0.pt',map_location='cpu',weights_only=False)
assert math.isclose(steps[start+1]['opd/lr'],initial['lr_scheduler']['_last_lr'][0],rel_tol=1e-8)
for step, m in steps.items():
    assert m['rollout/generated_sequences'] == 128 and m['tip/global_rollouts'] == 128
    assert m['tip/global_normalization'] == 1 and 0.48 < m['tip/selected_fraction'] <= 0.5
    assert math.isfinite(m['opd/loss']) and math.isfinite(m['opd/grad_norm']) and m['opd/grad_norm'] > 0
outputs = list((run/'resume-smoke-output').glob('Qwen3-4B-*'))
assert len(outputs)==1
out=outputs[0]
for step in steps:
    for rank in range(8):
        for kind in ('model','optim','extra_state'):
            f=out/f'global_step_{step}'/'actor'/f'{kind}_world_size_8_rank_{rank}.pt'
            assert f.is_file() and f.stat().st_size>0,f
    extra=torch.load(out/f'global_step_{step}'/'actor'/'extra_state_world_size_8_rank_0.pt',map_location='cpu',weights_only=False)
    assert extra['lr_scheduler']['last_epoch']==step
    loader = torch.load(out/f'global_step_{step}'/'data.pt', map_location='cpu', weights_only=False)
    main = loader['_snapshot']['_main_snapshot']
    assert main['_sampler_iter_state']['samples_yielded'] == step * 8, main
    assert loader['_snapshot']['_snapshot_step'] + loader['_steps_since_snapshot'] == step

assert (out/'latest_checkpointed_iteration.txt').read_text().strip()==str(start+2)
validation=list((out/f'{start+2}.jsonl').open())
assert len(validation)==32
report={'passed':True,'source':str(source),'steps':steps,'output':str(out),'validation_sequences':32}
(run/'resume-smoke-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Resume smoke PASS: model/optimizer/scheduler/data restoration, two updates, checkpoints, 32 validation outputs')
