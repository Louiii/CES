import pytest
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.optimized_inference import OptimizedCES
from ces.workflows.ces_cache_timing import make_cache_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA inference test')


def small_model(style='split', heads=2, dual=True):
    torch.manual_seed(123)
    m=GPT(GPTConfig(n_layer=4,n_head=2,n_kv_head=heads,n_embd=128,
                   vocab_size=256,sequence_len=64,window_pattern='SL',
                   dual_rope_style=style,use_dual_rope=dual)).cuda()
    m.init_weights()
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim>=2:p.normal_(0,.025)
        m.smear_lambda.fill_(.7);m.x0_lambdas.fill_(.1)
        for block in m.transformer.h:
            if block.attn.append_pointer_rope:block.attn.pointer_attn_gate.fill_(.3)
    return m.eval()


@pytest.mark.parametrize('style,heads,dual',[('split',2,True),('split',1,True),('append',2,True),('split',2,False)])
@torch.inference_mode()
def test_logits_addresses_and_reset(style,heads,dual):
    m=small_model(style,heads,dual)
    before={n:p.clone() for n,p in m.named_parameters()}
    opt=OptimizedCES(m,64)
    for ids,addresses in [([3,4,5,6,7,8],[0,1,2,0,1,5]),([9],[0])]:
        cache=make_cache_pair(m,capacity=64,device=torch.device('cuda'))
        x=torch.tensor([ids],device='cuda');a=torch.tensor([addresses],device='cuda')
        vl,pl=m(x,tix=a,kv_cache=cache.model,pos_k_cache=cache.position,return_pos_logits=True)
        torch.testing.assert_close(opt.prefill(ids,addresses),vl[:,-1],rtol=.02,atol=.025)
        torch.testing.assert_close(opt.address_logits(),pl,rtol=.03,atol=.04)
        for j in range(5):
            token=11+j;address=j//2
            xx=torch.tensor([[token]],device='cuda');aa=torch.tensor([[address]],device='cuda')
            vl,pl=m(xx,tix=aa,kv_cache=cache.model,pos_k_cache=cache.position,return_pos_logits=True)
            torch.testing.assert_close(opt.step(token,address),vl[:,-1],rtol=.02,atol=.025)
            torch.testing.assert_close(opt.address_logits(),pl,rtol=.03,atol=.04)
    for n,p in m.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    expected=(opt.k.numel()+opt.v.numel()+opt.address_keys.numel())*2
    assert opt.cache_bytes==expected


@torch.inference_mode()
def test_cuda_graph_replay_and_capacity():
    m=small_model();e=OptimizedCES(m,32)
    e.static_attention=True  # compare graph replay with identical attention kernels
    e.prefill([1,4,7]);reference=e.generate(12).clone();logits=e.logits.clone()
    e.prepare_backend('cuda_graph');e.prefill([1,4,7]);actual=e.generate(12)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    torch.testing.assert_close(e.logits,logits,rtol=0,atol=0)
    # Replay after a shorter prompt, and an explicit non-current pointer address.
    e.prefill([2]);e.step(8,0)
    cached=e.address_logits().clone()
    e.backend='eager';e.prefill([2]);e.step(8,0)
    torch.testing.assert_close(e.address_logits(),cached,rtol=0,atol=0)
    with pytest.raises(ValueError):e.generate(32)
    with pytest.raises(ValueError):e.step(5,33)
    with pytest.raises(ValueError):e.prefill([1,2],[0,2])
    with pytest.raises(ValueError):e.set_banned_tokens(range(256))
    e.set_banned_tokens(range(255));e.prefill([2]);assert e.generate(4).tolist()==[[255]*4]


@torch.inference_mode()
def test_small_capacity_and_mask_update():
    e=OptimizedCES(small_model(),2)
    e.prepare_backend('cuda_graph')
    e.prefill([2])
    e.set_banned_tokens(range(255))
    assert e.generate(1).tolist()==[[255]]
    with pytest.raises(ValueError):e.generate(1)
    one=OptimizedCES(small_model(),1)
    one.prefill([2])
    with pytest.raises(ValueError):one.prepare_backend('cuda_graph')
