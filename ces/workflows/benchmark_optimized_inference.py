"""Compare opt-in CES inference to the untouched cache benchmark on one GPU.

Writes raw repeat timings and correctness diagnostics. No model/data downloads,
training, checkpoint writes, or manuscript changes are performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from nanochat.gpt import GPT,GPTConfig
from nanochat.checkpoint_manager import _patch_missing_config_keys,_patch_missing_keys
from nanochat.optimized_inference import OptimizedCES
from ces.workflows.ces_cache_timing import make_cache_pair,_prefill,_resume_decode


def load_model(directory,step):
    meta=json.loads((directory/f'meta_{step:06d}.json').read_text())
    config=dict(meta['model_config']);_patch_missing_config_keys(config)
    state=torch.load(directory/f'model_{step:06d}.pt',map_location='cpu',mmap=True,weights_only=True)
    state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
    cfg=GPTConfig(**config);_patch_missing_keys(state,cfg)
    with torch.device('meta'):m=GPT(cfg)
    m.to_empty(device='cuda');m.init_weights();m.load_state_dict(state,assign=False)
    # Preserve checkpoint dtypes, just as build_model(assign=True) does.
    for name,p in m.named_parameters():
        if p.dtype!=state[name].dtype:p.data=p.data.to(state[name].dtype)
    return m.eval().requires_grad_(False),meta


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000,result


def diagnostics(actual,expected):
    a=actual.float();e=expected.float();delta=(a-e).abs()
    return {'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),
            'rms':float((delta.square().mean()).sqrt()),
            'argmax_match':bool(torch.equal(a.argmax(-1),e.argmax(-1)))}


@torch.inference_mode()
def correctness(model,engine,cases,backend):
    reports=[]
    for case in cases:
        tokens=case['source_tokens']
        addresses=list(range(len(tokens)))
        # Exercise non-current addresses too, not only plain AR input.
        addresses[-3:]=[1,2,3]
        caches=make_cache_pair(model,capacity=engine.capacity,device=torch.device('cuda'))
        ids=torch.tensor([tokens],device='cuda');a=torch.tensor([addresses],device='cuda')
        vl,pl=model(ids,tix=a,kv_cache=caches.model,pos_k_cache=caches.position,return_pos_logits=True)
        ev=engine.prefill(tokens,addresses)
        steps=[{'vocab':diagnostics(ev,vl[:,-1]),'address':diagnostics(engine.address_logits(),pl)}]
        for j in range(16):
            token=tokens[8+j]
            address=j if j%3==0 else len(tokens)+j
            vl,pl=model(torch.tensor([[token]],device='cuda'),tix=torch.tensor([[address]],device='cuda'),
                        kv_cache=caches.model,pos_k_cache=caches.position,return_pos_logits=True)
            ev=engine.step(token,address)
            steps.append({'vocab':diagnostics(ev,vl[:,-1]),'address':diagnostics(engine.address_logits(),pl)})
        reports.append({'backend':backend,'case_id':case['case_id'],'steps':steps})
        del caches
    return reports


@torch.inference_mode()
def greedy_agreement(model, engine, cases, banned, count=32):
    """Compare free-running continuations, separately from forced-input logits."""
    reports=[]
    for case in cases:
        for arm in ('source_tokens', 'edited_tokens'):
            tokens=case[arm]
            caches=make_cache_pair(model,capacity=engine.capacity,device=torch.device('cuda'))
            logits=_prefill(model,caches,tokens)
            generated=[]
            for j in range(count):
                masked=logits.clone();masked[:,banned]=float('-inf')
                token=masked.argmax(-1,keepdim=True)
                generated.append(token.clone())
                address=torch.full_like(token,len(tokens)+j)
                logits,_=model(token,tix=address,kv_cache=caches.model,
                    pos_k_cache=caches.position,return_pos_logits=True)
                logits=logits[:,-1]
            original=torch.cat(generated,dim=1)
            engine.prefill(tokens);optimized=engine.generate(count)
            equal=(original==optimized).flatten().cpu().tolist()
            reports.append({'case_id':case['case_id'],'arm':arm,
                'exact_sequence_match':all(equal),
                'first_divergence':next((i for i,v in enumerate(equal) if not v),None),
                'matching_tokens':sum(equal),'count':count,
                'legacy_tokens':original.cpu().tolist()[0],
                'optimized_tokens':optimized.cpu().tolist()[0]})
            del caches
    return reports


@torch.inference_mode()
def measure(model,engine,case,count,backend,warmups,repeats,banned):
    src,edited=case['source_tokens'],case['edited_tokens']
    if max(len(src),len(edited))+count>engine.capacity:return []
    # Each iteration starts with fresh logical cache state. Buffers stay allocated.
    if backend=='legacy':
        caches=make_cache_pair(model,capacity=engine.capacity,device=torch.device('cuda'))
        def prefill(tokens):
            caches.model.reset();caches.position.reset()
            return _prefill(model,caches,tokens)
        def resume(logits):
            return _resume_decode(model,caches,logits,
                                  num_tokens=count,banned_token_ids=banned)
    else:
        prefill=engine.prefill
        def resume(logits):return engine.generate(count)
    records=[]
    for rep in range(-warmups,repeats):
        # Alternate order to reduce systematic drift; independent prefill per arm.
        timings={}
        order=['source','edited'] if rep%2==0 else ['edited','source']
        for arm in order:
            pref_ms,logits=timed(lambda:prefill(src if arm=='source' else edited))
            decode_ms,_=timed(lambda:resume(logits))
            timings[arm]=(pref_ms,decode_ms)
        if rep>=0:
            records.append({'case_id':case['case_id'],'source_tokens':len(src),'resume_tokens':count,
                'backend':backend,'repeat':rep,'source_prefill_ms':timings['source'][0],
                'cached_resume_ms':timings['source'][1],'rebuild_ms':timings['edited'][0],
                'rebuilt_resume_ms':timings['edited'][1],
                'rebuild_plus_resume_ms':sum(timings['edited'])})
    return records


def summarise(records):
    summaries=[]
    for backend,count in sorted({(r['backend'],r['resume_tokens']) for r in records}):
        group=[r for r in records if r['backend']==backend and r['resume_tokens']==count]
        ids=sorted({r['case_id'] for r in group})
        row={'backend':backend,'resume_tokens':count,'cases':len(ids)}
        metrics=['source_prefill_ms','cached_resume_ms','rebuild_ms','rebuilt_resume_ms','rebuild_plus_resume_ms']
        per_case=[]
        for cid in ids:
            vals=[r for r in group if r['case_id']==cid]
            per_case.append({key:statistics.median(r[key] for r in vals) for key in metrics})
        for key in metrics:row[key]=statistics.median(r[key] for r in per_case)
        ratios=np.array([r['rebuild_plus_resume_ms']/r['cached_resume_ms'] for r in per_case])
        rng=np.random.default_rng(20261002)
        boots=ratios[rng.integers(0,len(ratios),(5000,len(ratios)))].mean(axis=1)
        row['mean_paired_ratio']=float(ratios.mean());row['ratio_ci95']=np.quantile(boots,[.025,.975]).tolist()
        row['tokens_per_second']=1000*count/row['cached_resume_ms']
        summaries.append(row)
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--step',type=int,default=74800)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backends',nargs='+',choices=['legacy','eager','cuda_graph','compile','compile_cuda_graph'],default=['legacy','eager','cuda_graph'])
    p.add_argument('--resume',type=int,nargs='+',default=[32,128,256])
    p.add_argument('--warmups',type=int,default=2);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--quick',action='store_true');p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    all_cases=[json.loads(s) for s in args.pairs.read_text().splitlines() if s]
    # Nine fixed cases: three lengths crossed with three paired location/mass settings.
    wanted={f'tokens{length}_loc{loc}_mass{mass}' for length in [128,512,1792]
            for loc,mass in [('0p10',1),('0p50',16),('0p90',64)]}
    cases=[r for r in all_cases if r['case_id'] in wanted]
    assert len(cases)==9,[r['case_id'] for r in cases]
    if args.quick:cases=[cases[0]]
    print('Loading d24 checkpoint',flush=True)
    model,meta=load_model(args.checkpoint,args.step)
    engine=OptimizedCES(model,model.config.sequence_len)
    # These benchmarks are ordinary continuation: retain the legacy control mask.
    from nanochat.tokenizer import get_extended_indel_tokeniser
    from nanochat.causal_serialisation_extended import INDEL_SPECIALS
    tokenizer=get_extended_indel_tokeniser()
    banned=sorted({tokenizer.get_bos_token_id(),*(tokenizer.encode_special(s) for s in INDEL_SPECIALS
                   if 0 <= tokenizer.encode_special(s) < model.config.vocab_size)})
    engine.set_banned_tokens(banned)
    evidence={'checkpoint':str(args.checkpoint),'step':args.step,'model_config':meta['model_config'],
              'torch':torch.__version__,'device':torch.cuda.get_device_name(),
              'numerical_guardrails':{'eager_max_abs':.05,'static_max_abs':.5,'max_rms':.2},
              'pairs_sha256':hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
              'case_ids':[c['case_id'] for c in cases],'capacity':engine.capacity,
              'cache_bytes':engine.cache_bytes,
              'legacy_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
              'optimized_parameter_bytes':sum(p.numel()*p.element_size() for p in engine.model.parameters()),
              'warmups':args.warmups,'repeats':args.repeats,
              'banned_token_ids':banned,'correctness':[],'backend_setup_seconds':{},
              'protocol':'Synchronized wall time; preallocated caches; compile/capture excluded. Prefill includes input construction. Same fixed-length greedy policy and banned tokens. Source prefill precedes cached-resume timing and is excluded from it.'}
    records=[]
    def save():
        evidence['summary']=summarise(records)
        evidence['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        (args.output/'summary.json').write_text(json.dumps(evidence,indent=2)+'\n')
        (args.output/'raw_timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    for backend in args.backends:
        print('Backend',backend,flush=True)
        if backend!='legacy':
            start=time.perf_counter();engine.prepare_backend(backend)
            evidence['backend_setup_seconds'][backend]=time.perf_counter()-start
            validation_cases=[cases[i] for i in range(0,len(cases),3)]
            result=correctness(model,engine,validation_cases,backend)
            evidence['correctness'].extend(result)
            maxerr=max(s['vocab']['max_abs'] for r in result for s in r['steps'])
            matches=sum(s['vocab']['argmax_match'] for r in result for s in r['steps'])
            total=sum(len(r['steps']) for r in result)
            print('Correctness vocab max_abs',maxerr,'argmax matches',matches,'/',total,flush=True)
            # Guard against silent semantic drift. Small BF16 kernel differences are recorded.
            # Static full-capacity attention selects a different BF16 SDPA
            # kernel from native prefix views; the diagnostic isolates this.
            rms=max(s['vocab']['rms'] for r in result for s in r['steps'])
            limit=.05 if backend=='eager' else .5
            if maxerr>limit or rms>.2:
                save();raise RuntimeError('Numerical deviation exceeds benchmark guardrail')
            evidence.setdefault('greedy_agreement',{})[backend]=greedy_agreement(
                model,engine,validation_cases,banned)
            if backend in ('cuda_graph','compile_cuda_graph'):
                engine.backend='eager';engine.prefill(cases[0]['source_tokens']);eager_tokens=engine.generate(32).clone()
                engine.backend=backend;engine.prefill(cases[0]['source_tokens']);graph_tokens=engine.generate(32)
                evidence['graph_static_eager_exact_tokens']=bool(torch.equal(eager_tokens,graph_tokens))
                if backend=='cuda_graph':assert evidence['graph_static_eager_exact_tokens']
            save()
        if args.validate_only:continue
        for count in args.resume:
            for case in cases:
                records.extend(measure(model,engine,case,count,backend,args.warmups,args.repeats,banned))
                save()
                print(backend,count,case['case_id'],'done',flush=True)
        print(json.dumps([r for r in summarise(records) if r['backend']==backend]),flush=True)
    save()


if __name__=='__main__':main()
