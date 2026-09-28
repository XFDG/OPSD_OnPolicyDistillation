"""Check real end-to-end TIP smoke evidence without loading model weights."""
import json
import math
from pathlib import Path
import re
import sys
import torch

run = Path(sys.argv[1])
log = re.sub(r"\x1b\[[0-9;]*m", "", (run / "smoke-train.log").read_text(errors="replace"))
live = run / 'ray-live.log'
if live.exists(): log += '\n' + live.read_text(errors='replace')
steps = {}
for line in log.splitlines():
    match = re.search(r"step:(\d+) - ", line)
    if match:
        values = dict(re.findall(r"([a-zA-Z0-9_/@-]+):([-+0-9.eE]+)", line))
        steps[int(match[1])] = {key: float(value) for key, value in values.items()}
assert set(steps) == {1, 2}, f"Expected two completed updates: {steps.keys()}"
for step, metrics in steps.items():
    for key, expected in {"rollout/prompts": 8, "rollout/samples_per_prompt": 16,
                          "rollout/generated_sequences": 128, "tip/global_rollouts": 128,
                          "tip/enabled": 1, "tip/global_normalization": 1}.items():
        assert metrics.get(key) == expected, (step, key, metrics.get(key))
    assert 0.48 < metrics["tip/selected_fraction"] <= 0.5
    assert math.isfinite(metrics["opd/loss"]) and metrics["opd/loss"] >= 0
    assert math.isfinite(metrics["opd/grad_norm"]) and metrics["opd/grad_norm"] > 0
assert math.isclose(steps[1]["opd/lr"], 1e-6, rel_tol=1e-5)
assert math.isclose(steps[2]["opd/lr"], 5e-7, rel_tol=1e-5), steps[2]["opd/lr"]
outputs = list((run / "smoke-output").glob("Qwen3-4B-*"))
assert len(outputs) == 1
out = outputs[0]
for step in (1, 2):
    checkpoint = out / f"global_step_{step}"
    assert (checkpoint / "data.pt").is_file()
    for rank in range(8):
        for kind in ("model", "optim", "extra_state"):
            file = checkpoint / "actor" / f"{kind}_world_size_8_rank_{rank}.pt"
            assert file.is_file() and file.stat().st_size > 0, file
        extra = torch.load(checkpoint / "actor" / f"extra_state_world_size_8_rank_{rank}.pt", map_location="cpu", weights_only=False)
        assert extra['lr_scheduler']['last_epoch'] == step
        assert 'rng' in extra
    state = torch.load(checkpoint / 'data.pt', map_location='cpu', weights_only=False)
    assert state['_snapshot']['_snapshot_step'] + state['_steps_since_snapshot'] == step
    assert state['_snapshot']['_main_snapshot']['_sampler_iter_state']['samples_yielded'] == step * 8
assert (out / "latest_checkpointed_iteration.txt").read_text().strip() == "2"
validation = [json.loads(line) for line in (out / "2.jsonl").open()]
assert len(validation) == 32, len(validation)
report = {"passed": True, "steps": steps, "validation_sequences": len(validation),
          "checkpoint_steps": [1, 2], "output": str(out)}
(run / "smoke-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print("TIP smoke: two updates, 128 rollouts/update, global Soft-OR, finite gradients, cosine LR, two eight-rank checkpoints and 32 validation outputs: PASS")
