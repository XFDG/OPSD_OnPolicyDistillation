"""Bounded BF16 FlashAttention forward/backward check on all eight B200s."""
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from flash_attn import flash_attn_func
assert torch.cuda.device_count() == 8
for i in range(8):
    assert 'B200' in torch.cuda.get_device_name(i)
    torch.manual_seed(42)
    q,k,v = [torch.randn(2,128,8,128,device=f'cuda:{i}',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    out = flash_attn_func(q,k,v,causal=True)
    with sdpa_kernel(SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(q.float().transpose(1,2),k.float().transpose(1,2),v.float().transpose(1,2),is_causal=True).transpose(1,2)
    error = ((out.float()-ref).norm()/ref.norm()).item()
    assert error < 0.02, (i,error)
    out.float().square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q,k,v))
    torch.cuda.synchronize(i)
    print(f'[kernel gate] GPU {i} BF16 FA2 forward/backward PASS relative_L2={error:.6g}',flush=True)
    del q,k,v,out,ref
    torch.cuda.empty_cache()
