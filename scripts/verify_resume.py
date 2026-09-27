"""Fail closed before resuming: require all four shards and training state."""
import json
from pathlib import Path
import re
import sys
import torch
p = Path(sys.argv[1]).resolve()
assert str(p).startswith('/volume/pt-train/users/zhaoye/OPSD-runtime/runs/'), p
assert re.fullmatch(r'global_step_[0-9]+', p.name), p
step = int(p.name.rsplit('_', 1)[1])
for rank in range(4):
    for kind in ('model', 'optim', 'extra_state'):
        f = p/'actor'/f'{kind}_world_size_4_rank_{rank}.pt'
        assert f.is_file() and f.stat().st_size > 0, f
    extra = torch.load(p/'actor'/f'extra_state_world_size_4_rank_{rank}.pt', map_location='cpu', weights_only=False)
    assert extra['lr_scheduler']['last_epoch'] == step
    assert 'rng' in extra
loader = torch.load(p/'data.pt', map_location='cpu', weights_only=False)
assert loader and '_snapshot' in loader
report = {'checkpoint': str(p), 'global_step': step, 'world_size': 4,
          'next_lr': extra['lr_scheduler']['_last_lr'][0], 'passed': True}
if len(sys.argv) > 2:
    (Path(sys.argv[2])/'resume-source.json').write_text(json.dumps(report, indent=2)+'\n')
print('Resume checkpoint PASS:', json.dumps(report))
