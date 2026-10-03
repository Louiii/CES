"""Opt-in, batch-one CES inference. Does not modify GPT or the legacy engine.

A private inference snapshot caches compute-dtype matrix weights. Static K/V
buffers, device-side indices and masked SDPA make the one-token step CUDA-graph
compatible. The address cache holds keys only; scoring addresses is explicit.
This is a model/cache API, not a replacement for the CES grammar/resolver.
"""
from __future__ import annotations
import copy
import math
import torch
import torch.nn.functional as F
from nanochat.gpt import Linear, norm, apply_rotary_emb, compute_cos_sin
from nanochat.common import COMPUTE_DTYPE


class OptimizedCES:
    def __init__(self, model, capacity=None):
        if model.training:
            raise ValueError('Pass an eval-mode model')
        self.capacity = model.config.sequence_len if capacity is None else int(capacity)
        if not 1 <= self.capacity <= model.config.sequence_len:
            raise ValueError('capacity must fit the trained context')
        # Never recast or monkeypatch the caller's training/inference model.
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.device = model.get_device()
        self.dtype = COMPUTE_DTYPE
        for module in self.model.modules():
            if isinstance(module, (Linear, torch.nn.Embedding)):
                module.to(dtype=self.dtype)
        c = model.config
        self.head_dim = model.get_kv_cache_head_dim()
        shape = (c.n_layer, 1, self.capacity, c.n_kv_head, self.head_dim)
        self.k = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v = torch.zeros_like(self.k)
        self.address_keys = (torch.zeros(1, self.capacity, c.n_embd, device=self.device,
                                        dtype=self.dtype) if model.position_head is not None else None)
        self.position = torch.zeros(1, device=self.device, dtype=torch.long)
        self.slots = torch.arange(self.capacity, device=self.device)
        self.previous_embedding = torch.zeros(1, 1, c.n_embd, device=self.device, dtype=self.dtype)
        self.last_hidden = torch.zeros_like(self.previous_embedding)
        self.input_token = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.input_address = torch.zeros_like(self.input_token)
        self.next_token = torch.zeros_like(self.input_token)
        self.logits = torch.zeros(1, c.vocab_size, device=self.device, dtype=torch.float32)
        self.allowed = torch.ones(c.vocab_size, device=self.device, dtype=torch.bool)
        # Precompute address rotations; source addresses are context positions.
        adim = self.model.half_head if self.model.dual_rope_style == 'split' else c.n_embd // c.n_head
        self.address_cos, self.address_sin = compute_cos_sin(
            self.slots, adim, model.rope_base, self.device, dtype=self.dtype)
        self.length = 0
        self.graph = None
        self.compiled_step = None
        self.backend = 'eager'
        self.static_attention = False

    @property
    def cache_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.k,self.v,self.address_keys) if t is not None)

    @torch.inference_mode()
    def reset(self):
        # Stale slots remain inaccessible under the attention/address masks.
        self.length = 0
        self.position.zero_()
        self.previous_embedding.zero_()
        self.last_hidden.zero_()

    @torch.inference_mode()
    def set_banned_tokens(self, ids):
        ids = list(map(int, ids))
        if any(i < 0 or i >= self.allowed.numel() for i in ids):
            raise ValueError('banned token outside vocabulary')
        if len(set(ids)) == self.allowed.numel():
            raise ValueError('at least one token must remain allowed')
        self.allowed.fill_(True)
        if ids:
            self.allowed[ids] = False
        if self.length:
            self.next_token.copy_(self.logits.masked_fill(~self.allowed, float('-inf')).argmax(-1, keepdim=True))

    def _attention(self, a, x, ve, cos_sin, address_rotations, positions, decode, layer):
        b,t,_ = x.shape
        q=a.c_q(x).view(b,t,a.n_head,a.head_dim)
        k=a.c_k(x).view(b,t,a.n_kv_head,a.head_dim)
        v=a.c_v(x).view(b,t,a.n_kv_head,a.head_dim)
        if ve is not None:
            ve=ve.view(b,t,a.n_kv_head,a.head_dim)
            gate=3*torch.sigmoid(a.ve_gate(x[...,:a.ve_gate_channels]))
            v=v+gate.unsqueeze(-1)*ve
        if a.append_pointer_rope:
            q=norm(apply_rotary_emb(q,*cos_sin))*1.2
            k=norm(apply_rotary_emb(k,*cos_sin))*1.2
            qp=a.c_q_pointer(x).view(b,t,a.n_head,a.head_dim)
            kp=a.c_k_pointer(x).view(b,t,a.n_kv_head,a.head_dim)
            qp=norm(apply_rotary_emb(qp,*address_rotations))*1.2
            kp=norm(apply_rotary_emb(kp,*address_rotations))*1.2
            q=torch.cat([q,a.pointer_attn_gate.to(q.dtype)*qp],dim=-1)
            k=torch.cat([k,kp],dim=-1)
            v=torch.cat([v,torch.zeros_like(v)],dim=-1)
            scale=a.head_dim**-.5
        elif a.use_dual_rope:
            q1,q2=q.chunk(2,dim=-1);k1,k2=k.chunk(2,dim=-1)
            q=torch.cat([apply_rotary_emb(q1,*cos_sin),apply_rotary_emb(q2,*address_rotations)],dim=-1)
            k=torch.cat([apply_rotary_emb(k1,*cos_sin),apply_rotary_emb(k2,*address_rotations)],dim=-1)
            q,k=norm(q)*1.2,norm(k)*1.2
            scale=None
        else:
            q=norm(apply_rotary_emb(q,*cos_sin))*1.2
            k=norm(apply_rotary_emb(k,*cos_sin))*1.2
            scale=None
        self.k[layer].index_copy_(1,positions,k)
        self.v[layer].index_copy_(1,positions,v)
        window=self.model.window_sizes[layer][0]
        if decode and not self.static_attention:
            # Host length is known without a device read. Preserve native SDPA's
            # valid-prefix view and kernel choice in the default eager backend.
            keys,values=self.k[layer,:,:self.length+1],self.v[layer,:,:self.length+1]
            if window>=0 and window<keys.shape[1]:
                keys,values=keys[:,-(window+1):],values[:,-(window+1):]
            mask=None;causal=False
        elif decode:
            keys,values=self.k[layer],self.v[layer]
            mask=self.slots[None,:]<=positions[:,None]
            if window>=0:
                mask=mask & ((positions[:,None]-self.slots[None,:])<=window)
            causal=False
        else:
            # Prefill starts at zero; attend only to actual prompt tokens.
            keys,values=k,v
            if window>=0 and window<t:
                mask=(positions[None,:]<=positions[:,None]) & ((positions[:,None]-positions[None,:])<=window)
                causal=False
            else:
                mask=None;causal=True
        y=F.scaled_dot_product_attention(q.transpose(1,2),keys.transpose(1,2),values.transpose(1,2),
            attn_mask=mask,is_causal=causal,enable_gqa=a.n_head!=a.n_kv_head,scale=scale)
        y=y.transpose(1,2)
        if a.append_pointer_rope:
            y=y[...,:a.head_dim]
        return a.c_proj(y.contiguous().view(b,t,-1))

    def _forward(self, ids, addresses, positions, decode):
        m=self.model
        cos_sin=(m.cos.index_select(1,positions),m.sin.index_select(1,positions))
        if m.use_dual_rope:
            rot=(self.address_cos[addresses].unsqueeze(2),self.address_sin[addresses].unsqueeze(2))
        else:
            rot=None
        x=norm(m.transformer.wte(ids).to(self.dtype))
        last_embedding=x[:,-1:,:].clone()
        if decode:
            gate=m.smear_lambda.to(x.dtype)*torch.sigmoid(m.smear_gate(x[:,:,:24]))
            x=x+gate*self.previous_embedding
        elif ids.shape[1]>1:
            gate=m.smear_lambda.to(x.dtype)*torch.sigmoid(m.smear_gate(x[:,1:,:24]))
            x=torch.cat([x[:,:1],x[:,1:]+gate*x[:,:-1]],dim=1)
        self.previous_embedding.copy_(last_embedding)
        x0=x
        backout=None
        for i,block in enumerate(m.transformer.h):
            x=m.resid_lambdas[i]*x+m.x0_lambdas[i]*x0
            ve=m.value_embeds[str(i)](ids).to(x.dtype) if str(i) in m.value_embeds else None
            x=x+self._attention(block.attn,norm(x),ve,cos_sin,rot,positions,decode,i)
            x=x+block.mlp(norm(x))
            if i==m.config.n_layer//2:
                backout=x
        x=norm(x-m.backout_lambda.to(x.dtype)*backout)
        self.last_hidden.copy_(x[:,-1:])
        if self.address_keys is not None:
            keys=norm(m.position_head.c_k(x))
            self.address_keys.index_copy_(1,positions,keys)
        logits=m.lm_head(x[:,-1:])[...,:m.config.vocab_size].float()
        return (15*torch.tanh(logits/15))[:,0,:]

    def _decode_step(self):
        logits=self._forward(self.input_token,self.input_address,self.position,True)
        self.logits.copy_(logits)
        self.next_token.copy_(logits.masked_fill(~self.allowed,float('-inf')).argmax(-1,keepdim=True))
        self.position.add_(1)

    @torch.inference_mode()
    def prefill(self, tokens, addresses=None):
        ids=torch.as_tensor(tokens,device=self.device,dtype=torch.long).reshape(1,-1)
        n=ids.shape[1]
        if not 1<=n<=self.capacity:
            raise ValueError('prompt length outside cache capacity')
        if bool(((ids<0)|(ids>=self.model.config.vocab_size)).any()):
            raise ValueError('token outside vocabulary')
        pos=torch.arange(n,device=self.device)
        addresses=pos[None,:] if addresses is None else torch.as_tensor(addresses,device=self.device,dtype=torch.long).reshape(1,-1)
        if addresses.shape!=ids.shape or bool(((addresses<0)|(addresses>pos[None,:])).any()):
            raise ValueError('addresses must refer to current or earlier positions')
        self.reset()
        self.logits.copy_(self._forward(ids,addresses,pos,False))
        self.next_token.copy_(self.logits.masked_fill(~self.allowed,float('-inf')).argmax(-1,keepdim=True))
        self.position.fill_(n);self.length=n
        return self.logits

    @torch.inference_mode()
    def step(self, token, address=None):
        if not 0<self.length<self.capacity:
            raise ValueError('prefill first; decoded token must fit cache')
        # Scalar inputs are host-known; tensor inputs must be trusted vocabulary IDs.
        if isinstance(token,int) and not 0<=token<self.model.config.vocab_size:
            raise ValueError('token outside vocabulary')
        if isinstance(address,int) and not 0<=address<=self.length:
            raise ValueError('address outside available context')
        self.input_token.copy_(torch.as_tensor(token,device=self.device).reshape(1,1))
        if address is None:
            self.input_address.copy_(self.position.reshape(1,1))
        else:
            self.input_address.copy_(torch.as_tensor(address,device=self.device).reshape(1,1))
        if self.backend in ('cuda_graph','compile_cuda_graph'):self.graph.replay()
        elif self.backend=='compile':self.compiled_step()
        else:self._decode_step()
        self.length+=1
        return self.logits

    @torch.inference_mode()
    def generate(self, count):
        """Fixed-length greedy continuation; no stop-token or grammar policy.

        Every emitted token is also ingested, matching ces_cache_timing.py.
        Output and state remain on device. Set banned controls before use.
        """
        if count<0 or self.length+count>self.capacity:
            raise ValueError('continuation exceeds cache capacity')
        output=torch.empty(1,count,device=self.device,dtype=torch.long)
        for i in range(count):
            output[:,i:i+1].copy_(self.next_token)
            self.step(self.next_token)
        return output

    @torch.inference_mode()
    def address_logits(self):
        """Score valid cached source addresses on demand, including current token."""
        if self.address_keys is None or self.length==0:
            raise ValueError('address head and non-empty cache required')
        q=norm(self.model.position_head.c_q(self.last_hidden))
        return (q@self.address_keys[:,:self.length].transpose(-1,-2))/math.sqrt(q.shape[-1])

    @torch.inference_mode()
    def prepare_backend(self, backend):
        """Compile/capture separately from measured work; invalidates current cache."""
        if backend not in ('eager','compile','cuda_graph','compile_cuda_graph'):
            raise ValueError('backend must be eager, compile, cuda_graph or compile_cuda_graph')
        if backend=='eager':
            self.backend=backend;self.static_attention=False
            return
        if self.device.type!='cuda':raise ValueError('accelerated backend requires CUDA')
        if self.capacity < 2:
            raise ValueError('accelerated decoding requires capacity of at least two')
        self.backend='eager'
        self.static_attention=True
        self.prefill([0]);self.input_token.zero_();self.input_address.fill_(1)
        if backend in ('compile','compile_cuda_graph'):
            self.compiled_step=torch.compile(self._decode_step,fullgraph=True,
                options={'triton.cudagraphs':False})
            self.compiled_step()
        if backend in ('cuda_graph','compile_cuda_graph'):
            operation=self.compiled_step if backend=='compile_cuda_graph' else self._decode_step
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.position.fill_(1);operation()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.position.fill_(1)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):operation()
        torch.cuda.synchronize(self.device)
        self.backend=backend
        self.reset()
