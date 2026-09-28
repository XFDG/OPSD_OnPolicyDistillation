"""Official FA4 paged GQA prefill/decode + CUDA graph versus FP32 attention."""
import json
from pathlib import Path
import sys
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from opd.fa4_adapter import official_varlen
import inspect
assert '/flash_attn/cute/' in inspect.getfile(official_varlen)
torch.manual_seed(20260928)
device='cuda:0'
page=128
k=torch.randn(9,page,4,128,device=device,dtype=torch.bfloat16)
v=torch.randn_like(k)
table=torch.tensor([[2,0,4],[3,1,5],[6,7,8]],device=device,dtype=torch.int32)
lengths=torch.tensor([97,133,255],device=device,dtype=torch.int32)
report=[]

def reference(q, qlens):
    pieces=[]; start=0
    for b,qlen in enumerate(qlens):
        length=int(lengths[b]); ids=table[b].long()
        kk=k[ids].flatten(0,1)[:length].float().repeat_interleave(4,dim=1)
        vv=v[ids].flatten(0,1)[:length].float().repeat_interleave(4,dim=1)
        qq=q[start:start+qlen].float();start+=qlen
        mask=torch.arange(length,device=device)[None,:] <= (torch.arange(qlen,device=device)[:,None]+length-qlen)
        with sdpa_kernel(SDPBackend.MATH):
            out=torch.nn.functional.scaled_dot_product_attention(qq.transpose(0,1)[None],kk.transpose(0,1)[None],vv.transpose(0,1)[None],attn_mask=mask)
        pieces.append(out[0].transpose(0,1))
    return torch.cat(pieces)

def check(name, output, q, qlens):
    ref=reference(q,qlens)
    diff=output.float()-ref
    rel=(diff.norm()/ref.norm()).item();maximum=diff.abs().max().item()
    assert torch.isfinite(output).all() and rel<0.01 and maximum<0.05,(name,rel,maximum)
    report.append(dict(case=name,relative_l2=rel,max_abs=maximum,passed=True))
    print(f'[FA4 gate] {name} PASS relative_L2={rel:.6g} max_abs={maximum:.6g}',flush=True)

def call(q,cu):
    return flash_attn_with_kvcache(q,k,v,page_table=table,cache_seqlens=lengths,cu_seqlens_q=cu,causal=True,ver=4)

with torch.inference_mode():
    qlens=[11,23,19]
    cu=torch.tensor([0,11,34,53],device=device,dtype=torch.int32)
    q=torch.randn(53,16,128,device=device,dtype=torch.bfloat16)
    check('paged_prefill_with_prefix',call(q,cu),q,qlens)
    cu=torch.arange(4,device=device,dtype=torch.int32)
    q=torch.randn(3,16,128,device=device,dtype=torch.bfloat16)
    check('paged_decode',call(q,cu),q,[1,1,1])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):call(q,cu)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=call(q,cu)
    lengths.copy_(torch.tensor([123,201,131],device=device,dtype=torch.int32))
    q.copy_(torch.randn_like(q))
    graph.replay();torch.cuda.synchronize()
    check('cuda_graph_changed_lengths_and_queries',out,q,[1,1,1])
Path(sys.argv[1]).write_text(json.dumps({'passed':True,'official_source':'e9cf2c1651d2303191eb40a739a3c135fda00999','implementation':inspect.getfile(official_varlen),'dtype':'bfloat16','cases':report},indent=2)+'\n')
