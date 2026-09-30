"""Build an opt-in resident teacher with the pinned Verl model builder.

Verl 0ddd28933f2d06fbf06d2d4b2cec7da547d596fc forces FSDP CPUOffload for
``role='ref'`` independently of ref.fsdp_config.param_offload. Merely turning
off the outer OPD load/offload calls therefore leaves per-forward parameter
transfers enabled.

For the supported FSDP1, full-model, non-QAT experiment, the parent's actor
construction path has identical model/dtype/wrapping inputs and disables that
internal offload. Passing optim_config=None prevents optimizer construction.
This helper changes only the builder's local role argument; the worker's role,
ref_module_fsdp assignment, eval/no_grad teacher forwards, and teacher weights
remain controlled by the original caller.
"""

import inspect

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.torch_dtypes import PrecisionType


def build_resident_ref(parent_build, worker, kwargs):
    """Return the parent's four-item build result after disabling ref offload.

    Fail closed for configuration changes that would make the actor builder
    perform additional work, such as QAT, LoRA, or FSDP2's offload policy.
    """
    call = dict(kwargs)
    inspect.signature(parent_build).bind(**call)
    if call.get("role") != "ref" or call.get("optim_config", object()) is not None:
        raise ValueError("Resident-ref construction requires role=ref and optim_config=None")
    if not worker._is_actor or not worker._is_ref:
        raise ValueError("Resident-ref construction requires the existing actor+ref worker")
    if worker.config.actor.strategy != "fsdp":
        raise ValueError("Resident-ref construction supports the pinned FSDP1 path only")
    if getattr(worker, "_qat_enabled", False):
        raise ValueError("Resident-ref construction does not support QAT")
    if (getattr(worker, "_is_lora", False)
            or worker.config.model.get("lora_rank", 0) > 0
            or worker.config.model.get("lora_adapter_path") is not None):
        raise ValueError("Resident-ref construction requires a full model without LoRA")

    fsdp_config = call["fsdp_config"]
    if fsdp_config.get("offload_policy", False) or fsdp_config.get("param_offload", False):
        raise ValueError("Resident-ref construction requires both ref offload settings disabled")
    if call.get("enable_gradient_checkpointing", False) or call.get("enable_activation_offload", False):
        raise ValueError("Resident-ref construction requires the original teacher memory settings")

    # The parent chooses the stored model dtype from this exact config and
    # worker._is_actor, not from its local role argument. Do not change either.
    dtype_name = fsdp_config.get("model_dtype")
    expected_dtype = PrecisionType.to_dtype(dtype_name) if dtype_name is not None else torch.float32
    original_flags = (worker.role, worker._is_actor, worker._is_ref, worker._is_rollout)
    call["role"] = "actor"
    result = parent_build(**call)
    if (worker.role, worker._is_actor, worker._is_ref, worker._is_rollout) != original_flags:
        raise RuntimeError("Resident-ref construction changed worker role flags")
    if len(result) != 4 or result[1] is not None or result[2] is not None:
        raise RuntimeError("Resident-ref construction unexpectedly created optimizer/scheduler state")
    model = result[0]
    if not isinstance(model, FSDP):
        raise RuntimeError("Resident-ref construction did not return FSDP1")
    wrappers = [module for module in model.modules() if isinstance(module, FSDP)]
    if any(module.cpu_offload.offload_params for module in wrappers):
        raise RuntimeError("Resident-ref construction retained internal FSDP CPU offload")
    parameters = list(model.parameters())
    if not parameters or any(parameter.device.type != "cuda" for parameter in parameters):
        raise RuntimeError("Resident-ref construction did not leave teacher parameters on CUDA")
    if any(parameter.dtype != expected_dtype for parameter in parameters):
        raise RuntimeError("Resident-ref construction changed the stored teacher parameter dtype")
    print(f"[opt] teacher resident: FSDP_wrappers={len(wrappers)} stored_dtype={expected_dtype} CUDA_params=True internal_CPU_offload=False", flush=True)
    return result
