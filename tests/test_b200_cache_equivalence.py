"""GPU check of actual TIP update: CPU versus GPU teacher-cache placement.

Full Qwen3 vocabulary, BF16 model/logits, AdamW; compare token statistics,
selection, loss, gradient, weights and optimizer moments exactly. No FSDP or
real-model claim: this isolates the changed cache placement semantics.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from omegaconf import OmegaConf
from verl import DataProto
from opd.losses import compute_tip_token_stats, select_tip_soft_or_indices
from opd.opd_worker import OPDWorker

torch.manual_seed(20260930)
vocab=151936
initial=torch.randn(32,vocab,device='cuda',dtype=torch.bfloat16)*0.1
teachers=[torch.randn(n,vocab,device='cuda',dtype=torch.bfloat16) for n in (12,9,6)]
ids=[torch.arange(n+1,device='cuda')[None] for n in (12,9,6)]
batches=[]
for x in ids:
    mask=torch.ones_like(x);mask[:,0]=0
    batches.append(DataProto.from_single_dict({'student_input_ids':x,
        'student_attention_mask':torch.ones_like(x),'student_position_ids':torch.arange(x.shape[1],device='cuda')[None],
        'student_loss_mask':mask,'valid_row_mask':torch.tensor([True],device='cuda')}))

def forward(net,x,attn,pos,mask):
    return net(x)[:,:-1][mask[:,1:].bool()]

results=[]
for location in ('cpu','cuda'):
    model=torch.nn.Embedding(32,vocab,device='cuda',dtype=torch.bfloat16)
    model.weight.data.copy_(initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=.1)
    cache=[(t.to(location),True) for t in teachers]
    worker=SimpleNamespace(actor_module_fsdp=model,ulysses_sequence_parallel_size=1,
        actor_optimizer=optimizer,config=OmegaConf.create({'actor':{'grad_clip':1.0},'tip':{'keep_ratio':.5,'entropy_clip_quantile':.98}}),
        _forward_logits_padded=forward)
    h=[];d=[]
    for b,(t,_) in zip(batches,cache):
        logits=forward(model,b.batch['student_input_ids'],None,None,b.batch['student_loss_mask'])
        hh,dd=compute_tip_token_stats(logits,t,512);h.append(hh.cpu());d.append(dd.cpu())
    selected,_=select_tip_soft_or_indices(torch.cat(h),torch.cat(d),[12,9,6],.5,.98)
    metrics=OPDWorker._tip_training_step(worker,batches,cache,device='cuda',batch_size=3,chunk_size=512)
    results.append({'stats':torch.stack((torch.cat(h),torch.cat(d))), 'selected':selected,'metrics':metrics,
        'weight':model.weight.detach().cpu().clone(),'grad':model.weight.grad.cpu().clone(),
        'optim':{k:v.cpu().clone() for k,v in optimizer.state[model.weight].items()}})
for key in ('stats','selected','weight','grad'):
    assert torch.equal(results[0][key],results[1][key]),key
assert results[0]['metrics']==results[1]['metrics']
for key in results[0]['optim']:
    assert torch.equal(results[0]['optim'][key],results[1]['optim'][key]),key
report={'passed':True,'vocab':vocab,'dtype':'bfloat16','exact_equal':['entropy','KL','selected_tokens','loss','gradient','weights','AdamW_state'],'scope':'actual TIP method with synthetic logits/embedding; single GPU, not distributed real-model equivalence'}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print('CPU/GPU teacher-cache equivalence: exact stats/selection/loss/gradient/weight/AdamW PASS',flush=True)
