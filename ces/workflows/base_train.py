"""
Train model. From root directory of the project, run as:

python -m ces.workflows.base_train

or distributed as:

torchrun --nproc_per_node=8 -m ces.workflows.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m ces.workflows.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import json
import time
import argparse
import queue
import threading
from collections import defaultdict
from dataclasses import asdict, replace
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import wandb
import torch.distributed as dist

from nanochat.causal_serialisation_extended import (
    CES_POINTER_TOKEN_MODES,
    DEL_OPEN,
    DEL_START,
    EDIT_START,
    INDL_BOS,
    INS_LOC,
    INS_OPEN,
    TokensOperations,
    TrainingTarget,
)
from nanochat.checkpoint_manager import (
    delete_checkpoint,
    load_checkpoint,
    load_tokenizer_for_checkpoint,
    save_checkpoint,
)
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    is_ddp_initialized,
    print0,
    print_banner,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.causal_serialisation import InsertTypeProbs
from nanochat.edit_dataloader import (
    EditSamplingConfig,
    deterministic_loader_mixture_with_state,
    tokenizing_distributed_partitioned_edit_loader_with_state,
    tokenizing_distributed_sequential_ar_dsl_loader_with_state,
    tokenizing_distributed_sequential_edit_loader_with_state,
    tokenizing_distributed_text_edit_baseline_loader_with_state,
)
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.loss_eval import evaluate_bpb, evaluate_ces_joint_bpb
from nanochat.pretrained_adapter import adapt_pretrained_state, new_token_ids
from nanochat.tokenizer import TOKENIZER_FACTORIES, get_token_bytes
from nanochat.tokenizer_contract import verify_canonical_tokenizer
from ces.workflows.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-seed", type=int, default=42, help="Global model-initialization/training RNG seed.")
parser.add_argument("--data-seed", type=int, default=1234, help="Training data-stream and edit-sampling seed.")
parser.add_argument(
    "--low-memory-distributed-optimizer",
    action="store_true",
    help="Process distributed optimizer groups sequentially to reduce peak VRAM at the cost of communication overlap.",
)
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
parser.add_argument(
    "--model-architecture",
    choices=("standard", "standard_locator", "ces_split", "ces_append"),
    default="ces_split",
    help="Single-RoPE GPT with optional locator head, legacy split-RoPE CES, or function-preserving appended-RoPE CES.",
)
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--stop-after-step", type=int, default=-1, help="Stop and checkpoint at this absolute step while retaining the full LR schedule horizon (-1 = full horizon).")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
parser.add_argument("--continue-to-data-matching", action="store_true", help="After reaching num_iterations (compute-matched), continue training until raw corpus tokens consumed matches baseline token-slot budget.")
parser.add_argument("--train-mode", choices=("clean", "plain_corrupt", "ces", "ar_dsl", "flat_relative", "regeneration", "fim", "fim_predicted"), default="ces", help="Training data mode: clean AR, corrupted AR, CES, AR-DSL, flat-relative coordinates, full regeneration, oracle-span FIM, or predicted-span FIM.")
parser.add_argument("--baseline-terminal-weight", type=float, default=1.0, help="Loss weight on the regeneration/FIM end marker; negative matches the completion-content length.")
parser.add_argument("--baseline-completion-only-loss", action=argparse.BooleanOptionalAction, default=False, help="For autoregressive edit DSL baselines, mask source-token loss and supervise only the complete edit program. Disabled by default.")
parser.add_argument("--baseline-single-document-pairs", action=argparse.BooleanOptionalAction, default=False, help="Build autoregressive/text edit pairs from one raw document rather than packed documents. Disabled by default.")
parser.add_argument("--edit-raw-data-dir", type=str, default=None, help="Raw parquet corpus directory for sequential edit loading.")
parser.add_argument("--edit-generated", nargs="*", default=(), help="Generated GPT edit parquet files or directories for sequential edit loading.")
parser.add_argument(
    "--edit-forward-direction-prob",
    type=float,
    default=0.15,
    help="Probability of training generated candidates source→target; the remainder trains inverse target→source edits.",
)
parser.add_argument("--edit-copy-scan-dir", type=str, default=None, help="Copy scan by_shard directory for sequential edit loading.")
parser.add_argument("--edit-random-corruption-prob", type=float, default=0.5, help="Probability of random fallback corruption on sequential rows with no sampled candidate edits.")
parser.add_argument("--edit-random-plain-corruption-prob", type=float, default=0.0, help="Probability of random corruption followed by plain AR training, for noisy-data baselines.")
parser.add_argument("--edit-noop-prob", type=float, default=0.0, help="Probability of serialising a no-op <edit-start><edit-end> example when no sampled candidate edits exist.")
parser.add_argument("--edit-clean-replay-prob", type=float, default=0.0, help="Fraction of CES training microbatches drawn from a separate clean/no-op replay corpus.")
parser.add_argument("--edit-clean-replay-raw-data-dir", type=str, default=None, help="Clean ClimbMix-style parquet directory used when --edit-clean-replay-prob is nonzero.")
parser.add_argument("--edit-clean-replay-noop-fraction", type=float, default=0.5, help="Within replay microbatches, fraction serialized as CES no-op programs; the remainder uses plain clean AR rows.")
parser.add_argument("--edit-random-replace-prob", type=float, default=0.32, help="Random fallback probability mass for replace before delete/insert split.")
parser.add_argument("--edit-random-delete-prob", type=float, default=0.315, help="Optional explicit random fallback probability mass for delete.")
parser.add_argument("--edit-random-insert-prob", type=float, default=0.315, help="Optional explicit random fallback probability mass for insert.")
parser.add_argument("--edit-random-move-prob", type=float, default=0.05, help="Optional explicit random fallback probability mass for move.")
parser.add_argument("--edit-random-del-mean-len", type=float, default=4.0, help="Random fallback deletion span geometric mean.")
parser.add_argument("--edit-random-ins-mean-len", type=float, default=4.0, help="Random fallback insertion payload geometric mean.")
parser.add_argument("--edit-random-no-del-mean", type=float, default=128.0, help="Random fallback mean gap before delete/replace.")
parser.add_argument("--edit-random-no-ins-mean", type=float, default=128.0, help="Random fallback mean gap before insert.")
parser.add_argument("--edit-random-source-token-reserve", type=int, default=128, help="Reserved serialisation budget for random fallback edits.")
parser.add_argument("--edit-random-source-token-reserve-max", type=int, default=-1, help="Maximum reserved suffix when jittering random-edit source length (-1 keeps it fixed).")
parser.add_argument("--edit-random-source-length-jitter-prob", type=float, default=0.0, help="Probability that a random edit uses a uniformly jittered source length.")
parser.add_argument("--edit-stream-clean-row-prob", type=float, default=0.0, help="Probability of a fully clean ordered-stream row; any nonzero stream mixture probability enables the ordered partition loader for CES.")
parser.add_argument("--edit-stream-partition-row-prob", type=float, default=0.0, help="Probability of a Gamma-partitioned multi-edit row; remaining non-clean probability uses one full-budget edit region.")
parser.add_argument("--edit-use-partitioned-loader", action=argparse.BooleanOptionalAction, default=False, help="Use the repaired ordered/partitioned CES loader even when clean and partition row probabilities are both zero.")
parser.add_argument("--edit-partition-min-size", type=int, default=48, help="Minimum Gamma-partition input length.")
parser.add_argument("--edit-partition-gamma-shape", type=float, default=2.0, help="Gamma shape for partition input lengths.")
parser.add_argument("--edit-partition-mean-size", type=float, default=360.0, help="Target mean Gamma-partition input length.")
parser.add_argument("--edit-priority-candidates", action=argparse.BooleanOptionalAction, default=True, help="Emit GPT/copy candidates consumed by background stream rows in dedicated full-context CES rows; use --no-edit-priority-candidates to disable.")
parser.add_argument("--edit-bos-aligned-row-packing", action=argparse.BooleanOptionalAction, default=True, help="Best-fit complete documents at physical row ends; split only true overlong documents.")
parser.add_argument("--edit-post-program-clean-suffix", action=argparse.BooleanOptionalAction, default=True, help="Place reserved clean continuation after edit-end instead of before the edit program.")
parser.add_argument("--edit-terminal-clean-suffix-min", type=int, default=64, help="Minimum reserved clean continuation tokens for edited regions.")
parser.add_argument("--edit-terminal-edit-end-prob", type=float, default=0.05, help="Explicit minority probability of placing edit-end at a region boundary.")
parser.add_argument("--edit-max-edits-per-example", type=int, default=4, help="Maximum sampled edits per sequential edit example.")
parser.add_argument("--edit-position-buffer-size", type=int, default=64, help="Padded position-query slots per example; affects CES addressing FLOPs.")
parser.add_argument("--edit-tokenizer-batch-size", type=int, default=128, help="Tokenizer/read batch size for sequential edit loader.")
parser.add_argument("--edit-reservoir-chunk-size", type=int, default=2048, help="Token chunk size for the live insertion-payload reservoir. The historical default is 2048; short structured corpora may opt into a smaller value.")
parser.add_argument("--edit-local-payload-only", action=argparse.BooleanOptionalAction, default=False, help="Sample random insertion/replacement payloads only from the current source or live corpus reservoir, avoiding full-vocabulary noise. Disabled by default.")
parser.add_argument("--loader-prefetch", type=int, default=2, help="Number of training batches to prefetch in a background thread (0 disables).")
parser.add_argument("--edit-corrupted-indel-bos-random-prob", type=float, default=0.7, help="For edited CES rows, probability of placing <|indel-bos|> uniformly before the earliest guarded edit address.")
parser.add_argument("--edit-corrupted-indel-bos-start-prob", type=float, default=0.3, help="For edited CES rows, probability of placing <|indel-bos|> at the start.")
parser.add_argument("--edit-clean-indel-bos-random-prob", type=float, default=0.5, help="For clean/noop CES rows, probability of placing <|indel-bos|> uniformly.")
parser.add_argument("--edit-clean-indel-bos-absent-prob", type=float, default=0.3, help="For clean/noop CES rows, probability of omitting <|indel-bos|> and training as plain AR.")
parser.add_argument("--edit-clean-indel-bos-start-prob", type=float, default=0.2, help="For clean/noop CES rows, probability of placing <|indel-bos|> at the start.")
parser.add_argument("--tokenizer", type=str, default="extended_indel", choices=sorted(TOKENIZER_FACTORIES), help="Tokenizer variant for base/edit training.")
parser.add_argument(
    "--ces-pointer-token-mode",
    choices=CES_POINTER_TOKEN_MODES,
    default="role_specific",
    help="CES pointer serialization: original role-specific tokens or one shared pointer token.",
)
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--weight-decay-schedule", choices=("cosine", "constant"), default="cosine", help="weight-decay schedule; constant is useful for shared stable trunks")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument("--resume-model-tag", type=str, default=None, help="load a resume checkpoint from another model tag while saving to --model-tag")
parser.add_argument("--init-from-model-tag", type=str, default=None, help="Import a standard checkpoint into a newly initialized target architecture.")
parser.add_argument("--init-from-step", type=int, default=-1, help="Source step for --init-from-model-tag.")
parser.add_argument("--init-new-token-seed", type=int, default=20260726, help="Deterministic seed for newly activated vocabulary rows.")
parser.add_argument("--adapt-new-only-steps", type=int, default=0, help="For an imported model, update only new modules/token rows for this many steps.")
parser.add_argument("--adapt-backbone-lr-scale", type=float, default=0.1, help="Joint-phase LR multiplier for imported backbone/shared vocabulary parameters.")
parser.add_argument("--resume-reset-optimizer", action="store_true", help="When resuming, load model/dataloader metadata but initialize a fresh optimizer instead of requiring optimizer shards.")
parser.add_argument(
    "--resume-reset-dataloader",
    action="store_true",
    help="Start the selected training corpus from its initial cursor instead of inheriting checkpoint loader state.",
)
parser.add_argument(
    "--delete-resume-checkpoint-after-load",
    action="store_true",
    help="Keep the loaded checkpoint as crash recovery, then delete it immediately before writing its replacement.",
)
parser.add_argument("--pos-warmup-ratio", type=float, default=0.01, help="ratio of iterations for position loss weight warmup")
parser.add_argument("--pos-weight-final", type=float, default=0.03, help="final position loss weight")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--ces-eval-tokens", type=int, default=8*524288, help="serialized validation tokens for deterministic CES joint metrics")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")






parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--save-optimizer", action="store_true", help="Save optimizer shards with checkpoints. Disabled by default for expensive experiment sweeps.")
parser.add_argument("--clean-example-match-target", type=int, default=-1, help="Continue past compute horizon until clean/raw examples seen reaches this global count (-1 disables).")
parser.add_argument("--grad-balance-every", type=int, default=100, help="log token/position gradient balance every N steps (-1 = disable)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
if args.low_memory_distributed_optimizer:
    # Triton pointwise autotuning clones large optimizer tensors while timing
    # candidate kernels. That transient copy can dominate peak VRAM for d24 on
    # 16 GiB cards, so use the deterministic default kernel in memory-first mode.
    from torch._inductor import config as inductor_config
    inductor_config.triton.autotune_pointwise = False
if args.edit_reservoir_chunk_size < 1:
    parser.error("--edit-reservoir-chunk-size must be positive")
if not 0.0 <= args.edit_forward_direction_prob <= 1.0:
    parser.error("--edit-forward-direction-prob must be in [0, 1]")
if args.resume_from_step != -1 and args.init_from_model_tag is not None:
    parser.error("Exact resume and pretrained import are separate modes; choose only one")
if (args.init_from_model_tag is None) != (args.init_from_step == -1):
    parser.error("--init-from-model-tag and --init-from-step must be provided together")
if args.adapt_new_only_steps < 0:
    parser.error("--adapt-new-only-steps must be non-negative")
if not 0.0 < args.adapt_backbone_lr_scale <= 1.0:
    parser.error("--adapt-backbone-lr-scale must be in (0, 1]")
standard_modes = {"clean", "ar_dsl", "flat_relative", "regeneration", "fim"}
if args.model_architecture == "standard" and args.train_mode not in standard_modes:
    parser.error("--model-architecture=standard supports clean and autoregressive edit baselines")
if args.model_architecture == "standard_locator" and args.train_mode != "fim_predicted":
    parser.error("--model-architecture=standard_locator is reserved for predicted-span FIM")
if args.train_mode == "fim_predicted" and args.model_architecture != "standard_locator":
    parser.error("--train-mode=fim_predicted requires --model-architecture=standard_locator")
if args.model_architecture == "standard" and args.train_mode == "clean" and args.tokenizer != "base":
    parser.error("Clean standard training requires --tokenizer=base")
if args.model_architecture == "standard" and args.train_mode == "ar_dsl" and args.tokenizer != "extended_indel":
    parser.error("Standard AR-DSL requires --tokenizer=extended_indel")
if args.model_architecture != "standard" and args.tokenizer == "base":
    parser.error("Editing architectures require an edit-capable tokenizer")
if args.train_mode == "flat_relative" and args.tokenizer != "flat_relative_2048":
    parser.error("--train-mode=flat_relative requires --tokenizer=flat_relative_2048")
if args.tokenizer == "flat_relative_2048" and args.train_mode != "flat_relative":
    parser.error("--tokenizer=flat_relative_2048 is reserved for --train-mode=flat_relative")
if args.ces_pointer_token_mode != "role_specific" and (
    args.train_mode != "ces" or args.tokenizer != "extended_indel"
):
    parser.error("--ces-pointer-token-mode=shared requires CES training with --tokenizer=extended_indel")
if args.init_from_model_tag is not None and args.model_architecture != "ces_append":
    parser.error("Pretrained adaptation requires --model-architecture=ces_append")
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type, seed=args.model_seed)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=COMPUTE_DTYPE)
    if device_type == "cuda" and COMPUTE_DTYPE in (torch.bfloat16, torch.float16)
    else nullcontext()
)
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

tokenizer_name = args.tokenizer
tokenizer = TOKENIZER_FACTORIES[tokenizer_name]()
if tokenizer_name == "extended_indel":
    tokenizer_contract = verify_canonical_tokenizer(tokenizer=tokenizer)
    user_config["tokenizer_contract"] = tokenizer_contract["name"]
    user_config["tokenizer_sha256"] = tokenizer_contract["tokenizer_sha256"]
    print0(
        f"Tokenizer contract: {tokenizer_contract['name']} "
        f"(extended vocab {tokenizer_contract['extended_vocab_size']:,})"
    )
user_config["tokenizer"] = tokenizer_name
try:
    if tokenizer_name == "extended_indel_char":
        token_bytes = torch.ones(tokenizer.get_vocab_size(), dtype=torch.int32, device=device)
        for token_id in tokenizer.enc._special_tokens.values():
            token_bytes[token_id] = 0
    else:
        token_bytes = get_token_bytes(device=device)
    if token_bytes.numel() < tokenizer.get_vocab_size():
        pad = torch.zeros(tokenizer.get_vocab_size() - token_bytes.numel(), dtype=token_bytes.dtype, device=device)
        token_bytes = torch.cat([token_bytes, pad])
    elif token_bytes.numel() > tokenizer.get_vocab_size():
        token_bytes = token_bytes[:tokenizer.get_vocab_size()]
except (AssertionError, FileNotFoundError):
    token_bytes = torch.zeros(tokenizer.get_vocab_size(), dtype=torch.int32, device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        use_dual_rope=args.model_architecture in {"ces_split", "ces_append"},
        dual_rope_style="append" if args.model_architecture == "ces_append" else "split",
        use_position_head=args.model_architecture != "standard" and args.train_mode != "flat_relative",
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
adapting = args.init_from_model_tag is not None
adaptation_active = adapting
initialization_metadata = None
adaptation_new_token_ids: list[int] = []
if resuming:
    resume_dirname = args.resume_model_tag or output_dirname
    resume_checkpoint_dir = os.path.join(base_dir, "base_checkpoints", resume_dirname)
    print0(f"Resuming optimization from {resume_dirname} step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        resume_checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=not args.resume_reset_optimizer,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    initialization_metadata = meta_data.get("initialization")
    if initialization_metadata and initialization_metadata.get("kind") == "pretrained_to_ces":
        adaptation_active = True
        saved_user_config = meta_data.get("user_config") or {}
        for policy_name, current_value in (
            ("adapt_new_only_steps", args.adapt_new_only_steps),
            ("adapt_backbone_lr_scale", args.adapt_backbone_lr_scale),
        ):
            saved_value = saved_user_config.get(policy_name)
            if saved_value is not None and saved_value != current_value:
                cli_name = policy_name.replace("_", "-")
                raise ValueError(
                    f"Adaptation resume policy mismatch for --{cli_name}: "
                    f"checkpoint={saved_value}, CLI={current_value}"
                )
        source_config = initialization_metadata.get("source_model_config") or {}
        source_vocab_size = source_config.get("vocab_size")
        if source_vocab_size is None:
            raise ValueError("Adaptation resume metadata is missing source vocabulary size")
        source_tokenizer = load_tokenizer_for_checkpoint(
            {"tokenizer": initialization_metadata.get("source_tokenizer")},
            source_vocab_size,
        )
        adaptation_new_token_ids = new_token_ids(source_tokenizer, tokenizer)
        print0(f"Continuing pretrained adaptation with {len(adaptation_new_token_ids)} new token rows")
    del model_data # free up this memory after the copy
elif adapting:
    source_checkpoint_dir = os.path.join(base_dir, "base_checkpoints", args.init_from_model_tag)
    print0(f"Importing pretrained model from {args.init_from_model_tag} step {args.init_from_step}")
    source_state, _source_optimizer, source_metadata = load_checkpoint(
        source_checkpoint_dir,
        args.init_from_step,
        device,
        load_optimizer=False,
        rank=ddp_rank,
    )
    source_tokenizer = load_tokenizer_for_checkpoint(
        source_metadata, source_metadata["model_config"]["vocab_size"]
    )
    initialization_metadata = adapt_pretrained_state(
        target_model=model,
        target_tokenizer=tokenizer,
        source_state=source_state,
        source_tokenizer=source_tokenizer,
        source_metadata=source_metadata,
        seed=args.init_new_token_seed,
    )
    adaptation_new_token_ids = new_token_ids(source_tokenizer, tokenizer)
    print0(
        f"Imported {initialization_metadata['shared_token_count']} shared tokens; "
        f"initialized {initialization_metadata['new_token_count']} new tokens"
    )
    del source_state

print0(f"TOKENIZER: {tokenizer.enc.name}")
print0(f"mergeable: {len(tokenizer.enc._mergeable_ranks)}")
print0(f"specials: {len(tokenizer.enc._special_tokens)}, max: {max(tokenizer.enc._special_tokens.values())}")
print0(f"computed_vocab: {max(max(tokenizer.enc._mergeable_ranks.values()), max(tokenizer.enc._special_tokens.values())) + 1}")
print0(f"model_vocab: {model_config.vocab_size}")
# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
backbone_flops_per_token = model.estimate_flops(position_queries=0)
num_flops_per_token = model.estimate_flops(position_queries=args.edit_position_buffer_size)
print0(f"Backbone-only FLOPs per token: {backbone_flops_per_token:e}")
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

data_match_raw_target = -1


# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    distributed_optimizer_overlap=not args.low_memory_distributed_optimizer,
)

if resuming and optimizer_data is not None:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data
elif resuming:
    print0("Resume reset optimizer enabled: continuing with a freshly initialized optimizer.")
for group in optimizer.param_groups:
    group["initial_weight_decay"] = group.get("weight_decay", 0.0)

delete_resume_checkpoint_before_save = resuming and args.delete_resume_checkpoint_after_load

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

token_ops = None
if args.model_architecture != "standard" or args.train_mode in {
    "ar_dsl",
    "flat_relative",
    "regeneration",
    "fim",
    "fim_predicted",
}:
    token_ops = TokensOperations(
        tokeniser=tokenizer.enc,
        pointer_token_mode=args.ces_pointer_token_mode,
    )
edit_sampling_config = EditSamplingConfig(
    raw_data_dir=args.edit_raw_data_dir,
    generated_edit_paths=tuple(args.edit_generated),
    copy_scan_dir=args.edit_copy_scan_dir,
    direction_probs={
        "forward": args.edit_forward_direction_prob,
        "inverse": 1.0 - args.edit_forward_direction_prob,
    },
    random_corruption_prob_when_no_candidates=args.edit_random_corruption_prob,
    random_plain_corruption_prob_when_no_candidates=args.edit_random_plain_corruption_prob,
    noop_edit_prob_when_no_candidates=args.edit_noop_prob,
    plain_corruption_only=args.train_mode == "plain_corrupt",
    random_replace_prob=args.edit_random_replace_prob,
    random_delete_prob=args.edit_random_delete_prob,
    random_insert_prob=args.edit_random_insert_prob,
    random_move_prob=args.edit_random_move_prob,
    random_del_mean_len=args.edit_random_del_mean_len,
    random_ins_mean_len=args.edit_random_ins_mean_len,
    random_no_del_mean=args.edit_random_no_del_mean,
    random_no_ins_mean=args.edit_random_no_ins_mean,
    random_source_token_reserve=args.edit_random_source_token_reserve,
    random_source_token_reserve_max=(
        None
        if args.edit_random_source_token_reserve_max < 0
        else args.edit_random_source_token_reserve_max
    ),
    random_source_length_jitter_prob=args.edit_random_source_length_jitter_prob,
    stream_clean_row_prob=args.edit_stream_clean_row_prob,
    stream_partition_row_prob=args.edit_stream_partition_row_prob,
    partition_min_size=args.edit_partition_min_size,
    partition_gamma_shape=args.edit_partition_gamma_shape,
    partition_slice_mean_size=args.edit_partition_mean_size,
    priority_candidate_lane=args.edit_priority_candidates,
    bos_aligned_row_packing=args.edit_bos_aligned_row_packing,
    post_edit_clean_suffix=args.edit_post_program_clean_suffix,
    terminal_clean_suffix_min=args.edit_terminal_clean_suffix_min,
    terminal_edit_end_prob=args.edit_terminal_edit_end_prob,
    random_insertion_type_probabilities=(
        InsertTypeProbs(buffer=0.85, myself=0.15, uniform=0.0, marginal=0.0)
        if args.edit_local_payload_only
        else InsertTypeProbs(buffer=0.6, myself=0.1, uniform=0.2, marginal=0.1)
    ),
    max_edits_per_example=args.edit_max_edits_per_example,
    max_serialised_tokens=args.max_seq_len,
    corrupted_indel_bos_random_prob=args.edit_corrupted_indel_bos_random_prob,
    corrupted_indel_bos_start_prob=args.edit_corrupted_indel_bos_start_prob,
    clean_indel_bos_random_prob=args.edit_clean_indel_bos_random_prob,
    clean_indel_bos_absent_prob=args.edit_clean_indel_bos_absent_prob,
    clean_indel_bos_start_prob=args.edit_clean_indel_bos_start_prob,
    seed=args.data_seed,
)
if args.train_mode == "clean":
    edit_sampling_config = replace(
        edit_sampling_config,
        generated_edit_paths=(),
        copy_scan_dir=None,
        random_corruption_prob_when_no_candidates=0.0,
        random_plain_corruption_prob_when_no_candidates=0.0,
        noop_edit_prob_when_no_candidates=0.0,
        random_source_token_reserve=0,
    )
elif args.train_mode == "plain_corrupt":
    edit_sampling_config = replace(
        edit_sampling_config,
        copy_scan_dir=None,
        random_corruption_prob_when_no_candidates=0.0,
        noop_edit_prob_when_no_candidates=0.0,
    )


# def get_pos_weight(step):
#     return 0.1

def get_pos_weight(step):
    warm = round(args.pos_warmup_ratio * num_iterations)  # add arg or reuse args.warmup_ratio
    if step < warm:
        return args.pos_weight_final * (step + 1) / warm
    else:
        return args.pos_weight_final


class PrefetchIterator:
    def __init__(self, iterator, max_prefetch: int):
        self.iterator = iterator
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.sentinel = object()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            try:
                item = next(self.iterator)
            except BaseException as exc:
                self.queue.put(exc)
                return
            self.queue.put(item)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item


# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = (
    None if not resuming or args.resume_reset_dataloader else meta_data["dataloader_state_dict"]
)
if args.edit_raw_data_dir is None:
    raise ValueError("--edit-raw-data-dir is required")
clean_replay_active = args.edit_clean_replay_prob > 0
if not 0.0 <= args.edit_clean_replay_prob < 1.0:
    raise ValueError("--edit-clean-replay-prob must be in [0, 1)")
if not 0.0 <= args.edit_clean_replay_noop_fraction <= 1.0:
    raise ValueError("--edit-clean-replay-noop-fraction must be in [0, 1]")
if clean_replay_active and args.train_mode != "ces":
    raise ValueError("--edit-clean-replay-prob is currently supported only for CES training")
if clean_replay_active and args.edit_clean_replay_raw_data_dir is None:
    raise ValueError(
        "--edit-clean-replay-raw-data-dir is required when clean replay is enabled"
    )
mixture_resume_state_dict = dataloader_resume_state_dict if clean_replay_active else None
primary_resume_state_dict = (
    mixture_resume_state_dict.get("primary_state")
    if mixture_resume_state_dict is not None
    else dataloader_resume_state_dict
)
replay_resume_state_dict = (
    mixture_resume_state_dict.get("replay_state")
    if mixture_resume_state_dict is not None
    else None
)
partitioned_layout_requested = (
    args.edit_use_partitioned_loader
    or edit_sampling_config.stream_clean_row_prob > 0
    or edit_sampling_config.stream_partition_row_prob > 0
)
ces_loader = (
    tokenizing_distributed_partitioned_edit_loader_with_state
    if args.train_mode == "ces" and partitioned_layout_requested
    else tokenizing_distributed_sequential_edit_loader_with_state
)


def standard_clean_loader_with_state(split, resume_state_dict=None):
    """Expose the native clean AR loader through the shared training-loop API."""

    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split=split,
        device=device,
        resume_state_dict=resume_state_dict,
        tokenizer_batch_size=args.edit_tokenizer_batch_size,
    )
    bos_token = tokenizer.get_bos_token_id()
    for inputs, tokens, state in loader:
        # The native best-fit loader reuses one on-device staging buffer. This
        # wrapper may run in PrefetchIterator's background thread, so hand the
        # training loop owned tensors before the producer advances and mutates
        # that buffer again.
        inputs = inputs.clone()
        tokens = tokens.clone()
        batch_size, sequence_len = inputs.shape
        positions = torch.arange(sequence_len, device=inputs.device).expand(batch_size, -1)
        target = TrainingTarget(
            tokens=tokens,
            gathered_pos_target=None,
            token_loss_weights=torch.ones_like(tokens, dtype=torch.bool),
            gather_indices=None,
            gather_valid=None,
            overflow=False,
            perplexity_masks=None,
        )
        state.update(
            raw_token_consumption=batch_size * sequence_len,
            raw_examples_consumption=int((inputs == bos_token).sum().item()),
            edit_mass=0,
            valid_pointer_targets=0,
        )
        yield (inputs, positions, target), state


if args.model_architecture == "standard" and args.train_mode == "clean":
    train_loader = standard_clean_loader_with_state("train", dataloader_resume_state_dict)
elif args.train_mode in {"clean", "plain_corrupt", "ces"}:
    train_loader = ces_loader(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train",
        cfg=edit_sampling_config, token_ops=token_ops, device=device,
        resume_state_dict=primary_resume_state_dict,
        tokenizer_batch_size=args.edit_tokenizer_batch_size,
        pos_buffer_size=args.edit_position_buffer_size,
        reservoir_chunk_size=args.edit_reservoir_chunk_size,
    )
    if clean_replay_active:
        replay_config = replace(
            edit_sampling_config,
            raw_data_dir=args.edit_clean_replay_raw_data_dir,
            generated_edit_paths=(),
            copy_scan_dir=None,
            random_corruption_prob_when_no_candidates=0.0,
            random_plain_corruption_prob_when_no_candidates=0.0,
            noop_edit_prob_when_no_candidates=0.0,
            random_source_token_reserve=0,
            stream_clean_row_prob=1.0 - args.edit_clean_replay_noop_fraction,
            stream_partition_row_prob=0.0,
            priority_candidate_lane=False,
            post_edit_clean_suffix=False,
            terminal_clean_suffix_min=0,
            terminal_edit_end_prob=1.0,
            seed=args.data_seed + 10_000,
        )
        replay_loader = tokenizing_distributed_partitioned_edit_loader_with_state(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            split="train",
            cfg=replay_config,
            token_ops=token_ops,
            device=device,
            resume_state_dict=replay_resume_state_dict,
            tokenizer_batch_size=args.edit_tokenizer_batch_size,
            pos_buffer_size=args.edit_position_buffer_size,
            reservoir_chunk_size=args.edit_reservoir_chunk_size,
        )
        train_loader = deterministic_loader_mixture_with_state(
            train_loader,
            replay_loader,
            args.edit_clean_replay_prob,
            resume_state_dict=mixture_resume_state_dict,
        )
        print0(
            f"Clean/no-op replay enabled: {args.edit_clean_replay_prob:.3f} of "
            f"microbatches ({args.edit_clean_replay_noop_fraction:.3f} no-op within replay) "
            f"from {args.edit_clean_replay_raw_data_dir}"
        )
elif args.train_mode in {"ar_dsl", "flat_relative"}:
    baseline_loader = (
        tokenizing_distributed_partitioned_edit_loader_with_state
        if partitioned_layout_requested
        else tokenizing_distributed_sequential_ar_dsl_loader_with_state
    )
    baseline_kwargs = dict(
        cfg=edit_sampling_config, token_ops=token_ops, device=device,
        resume_state_dict=dataloader_resume_state_dict,
        tokenizer_batch_size=args.edit_tokenizer_batch_size,
        pos_buffer_size=args.edit_position_buffer_size,
        reservoir_chunk_size=args.edit_reservoir_chunk_size,
        baseline_mode=args.train_mode,
    )
    if partitioned_layout_requested:
        if args.baseline_completion_only_loss:
            raise ValueError(
                "--baseline-completion-only-loss currently requires the sequential "
                "AR baseline loader; disable the ordered partition mixture"
            )
        train_loader = baseline_loader(
            tokenizer, args.device_batch_size, args.max_seq_len, split="train",
            **baseline_kwargs,
        )
    else:
        train_loader = baseline_loader(
            tokenizer, args.device_batch_size, args.max_seq_len, split="train",
            completion_only_loss=args.baseline_completion_only_loss,
            terminal_weight=args.baseline_terminal_weight,
            single_document_pairs=args.baseline_single_document_pairs,
            **baseline_kwargs,
        )
elif args.train_mode in {"regeneration", "fim", "fim_predicted"}:
    train_loader = tokenizing_distributed_text_edit_baseline_loader_with_state(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train",
        cfg=edit_sampling_config, token_ops=token_ops, device=device,
        resume_state_dict=dataloader_resume_state_dict,
        tokenizer_batch_size=args.edit_tokenizer_batch_size,
        pos_buffer_size=args.edit_position_buffer_size,
        baseline_mode=args.train_mode,
        baseline_terminal_weight=args.baseline_terminal_weight,
        reservoir_chunk_size=args.edit_reservoir_chunk_size,
        single_document_pairs=args.baseline_single_document_pairs,
    )
else:
    raise ValueError(f"Unsupported train mode: {args.train_mode}")
if args.loader_prefetch > 0:
    train_loader = PrefetchIterator(train_loader, args.loader_prefetch)
    print0(f"Training loader prefetch enabled: {args.loader_prefetch} batches")
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
build_ces_val_loader = lambda: ces_loader(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val",
    cfg=edit_sampling_config, token_ops=token_ops, device=device,
    resume_state_dict=None,
    tokenizer_batch_size=args.edit_tokenizer_batch_size,
    pos_buffer_size=args.edit_position_buffer_size,
    reservoir_chunk_size=args.edit_reservoir_chunk_size,
)
(x, tix, target), dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data
# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
stop_after_step = num_iterations if args.stop_after_step < 0 else args.stop_after_step
if not 0 < stop_after_step <= num_iterations:
    raise ValueError(
        f"--stop-after-step must be in [1, {num_iterations}], got {stop_after_step}"
    )
print0(f"This invocation will stop at absolute step {stop_after_step:,}")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# For data-matching: match the baseline "raw corpus token budget" to the compute horizon token-slot budget.
# Baseline consumes ~args.total_batch_size raw tokens per step, so baseline_raw_target == total_tokens.
data_match_raw_target = total_tokens if args.continue_to_data_matching else -1
clean_example_match_target = int(args.clean_example_match_target)
if args.continue_to_data_matching:
    print0(f"Data-matching enabled: will continue past compute horizon until raw_tokens_seen >= {data_match_raw_target:,}")
else:
    print0("Data-matching disabled: will stop at compute horizon (num_iterations).")
if clean_example_match_target > 0:
    print0(f"Clean-example matching enabled: will continue until clean_examples_seen >= {clean_example_match_target:,}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = min(max((num_iterations - it) / warmdown_iters, 0.0), 1.0)
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = min(max((it - warmdown_start) / warmdown_iters, 0.0), 1.0)
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    if args.weight_decay_schedule == "constant":
        return weight_decay_scaled
    it = min(max(it, 0), num_iterations)
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    smooth_pos_loss = 0
    total_training_time = 0 # total wall-clock time of training
    raw_tokens_seen = 0
    clean_examples_seen = 0
    edit_mass_seen = 0
    valid_pointer_targets_seen = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    smooth_pos_loss = 0  # HACK
    raw_tokens_seen = loop_state.get("raw_tokens_seen", 0)
    clean_examples_seen = loop_state.get("clean_examples_seen", 0)
    edit_mass_seen = loop_state.get("edit_mass_seen", 0)
    valid_pointer_targets_seen = loop_state.get("valid_pointer_targets_seen", 0)

if step > stop_after_step:
    raise ValueError(f"Resume step {step} is beyond requested stop step {stop_after_step}")
last_loader_wait = 0.0


def _is_vocab_parameter_name(name: str) -> bool:
    return name in {"transformer.wte.weight", "lm_head.weight"} or (
        name.startswith("value_embeds.") and name.endswith(".weight")
    )


def _is_new_parameter_name(name: str) -> bool:
    return (
        name.startswith("position_head.")
        or ".attn.c_q_pointer." in name
        or ".attn.c_k_pointer." in name
        or name.endswith(".attn.pointer_attn_gate")
    )


def apply_adaptation_gradient_policy(current_step: int) -> None:
    if not adaptation_active:
        return
    new_only = current_step < args.adapt_new_only_steps
    row_index = None
    if adaptation_new_token_ids:
        row_index = torch.tensor(adaptation_new_token_ids, dtype=torch.long, device=device)
    for name, parameter in orig_model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if _is_vocab_parameter_name(name):
            if row_index is None:
                grad.zero_()
                continue
            selected = grad.index_select(0, row_index).clone()
            if new_only:
                grad.zero_()
            else:
                # The whole vocabulary tensor uses the backbone LR. Boost the
                # new rows so their effective joint-phase LR remains unchanged.
                selected.div_(args.adapt_backbone_lr_scale)
            grad.index_copy_(0, row_index, selected)
        elif new_only and not _is_new_parameter_name(name):
            grad.zero_()


def debias(smooth, value, ema_beta: float = 0.9):
    # ema_beta: EMA decay factor for some smoothing just for nicer logging
    smooth = ema_beta * smooth + (1 - ema_beta) * value # EMA the training loss
    debiased = smooth / (1 - ema_beta**(step + 1)) # debias the EMA
    return smooth, debiased


def grad_norm_wrt(loss, probe, retain_graph=True):
    g = torch.autograd.grad(
        loss, probe,
        retain_graph=retain_graph,
        allow_unused=True
    )[0]
    return 0.0 if g is None else float(g.norm().detach().cpu())


def valid_count_from_batch(batch):
    _x, _tix, _target = batch
    gv = getattr(_target, "gather_valid", None)
    if gv is None:
        return 0
    return int(gv.sum().item())


def per_example_valid_counts(tgt: TrainingTarget):
    gv = tgt.gather_valid
    y  = tgt.gathered_pos_target
    if torch.is_tensor(gv):
        valid = gv.bool() & (y != -1)
        # valid shape: (B, G) -> sum over G
        return valid.sum(dim=1).detach().cpu().numpy()
    else:
        valid = gv.astype(bool) & (y != -1)
        return valid.sum(axis=1)


investigation_examples = []
min_valid_per_ex = 16
max_keep = 32  # examples, not batches

if args.grad_balance_every > 0 and args.train_mode == "ces":
    cpu_loader = ces_loader(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train",
        cfg=edit_sampling_config, token_ops=token_ops, device="cpu",
        resume_state_dict=primary_resume_state_dict,
        tokenizer_batch_size=args.edit_tokenizer_batch_size,
        reservoir_chunk_size=args.edit_reservoir_chunk_size,
    )

    for (batch, _meta) in cpu_loader:
        _x, _tix, _tgt = batch
        counts = per_example_valid_counts(_tgt)  # shape (B,)

        good = np.where(counts >= min_valid_per_ex)[0]
        for i in good:
            # store CPU tensors + index
            investigation_examples.append((batch, int(i), int(counts[i])))
            if len(investigation_examples) >= max_keep:
                break
        if len(investigation_examples) >= max_keep:
            break

    if not investigation_examples:
        raise ValueError(f"No investigation examples found with per-example valid >= {min_valid_per_ex}. "
                         f"Lower min_valid_per_ex or adjust corruption.")


def slice_example_from_batch(batch, i: int):
    _x, _tix, tgt = batch
    # slice batch dim i:i+1
    _x1 = _x[i:i+1]
    _tix1 = _tix[i:i+1]

    d = tgt._asdict()
    def slice0(v):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return v[i:i+1]
        return v

    for k in ["tokens", "gathered_pos_target", "token_loss_weights", "gather_indices", "gather_valid"]:
        d[k] = slice0(d[k])

    pm = d.get("perplexity_masks", None)
    if pm is not None:
        pm2 = {}
        for kk, mv in pm.items():
            if torch.is_tensor(mv) or isinstance(mv, np.ndarray):
                pm2[kk] = mv[i:i+1] if (mv.ndim >= 2) else mv
            else:
                pm2[kk] = mv
        d["perplexity_masks"] = pm2

    tgt1 = TrainingTarget(**d)
    return _x1, _tix1, tgt1


def move_to(obj, device, non_blocking=True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return {k: move_to(v, device, non_blocking) for k, v in obj.items()}

    # NamedTuple (like TrainingTarget): has _fields and _asdict()
    if hasattr(obj, "_fields") and hasattr(obj, "_asdict"):
        d = obj._asdict()
        d = {k: move_to(v, device, non_blocking) for k, v in d.items()}
        return type(obj)(**d)

    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to(x, device, non_blocking) for x in obj)

    return obj


def log_grad_balance(step, investigation_examples, max_examples=6):
    lambda_now = float(get_pos_weight(step))
    rows = []
    orig_model.eval()

    with torch.enable_grad():
        for (batch, i, valid_i) in investigation_examples[:max_examples]:
            _x, _tix, _target = slice_example_from_batch(batch, i)

            # Move to GPU
            _x = _x.to(device, non_blocking=True)
            _tix = _tix.to(device, non_blocking=True)
            _target = move_to(_target, device)

            # token grad pass
            orig_model.zero_grad(set_to_none=True)
            with autocast_ctx:
                _, aux = orig_model(_x, _tix, targets=_target, return_probe=True)
            probe = aux["probe"]
            loss_tok = aux["token_loss"]
            g_tok_t = torch.autograd.grad(loss_tok, probe, retain_graph=False, allow_unused=True)[0]
            g_tok = 0.0 if g_tok_t is None else float(g_tok_t.norm().detach().cpu())
            del aux, probe, loss_tok, g_tok_t
            torch.cuda.empty_cache()

            # pos grad pass
            orig_model.zero_grad(set_to_none=True)
            with autocast_ctx:
                _, aux = orig_model(_x, _tix, targets=_target, return_probe=True)
            probe = aux["probe"]
            loss_pos = aux["position_loss"]
            g_pos_t = torch.autograd.grad(loss_pos, probe, retain_graph=False, allow_unused=True)[0]
            g_pos = 0.0 if g_pos_t is None else float(g_pos_t.norm().detach().cpu())
            del aux, probe, loss_pos, g_pos_t
            torch.cuda.empty_cache()

            implied_r = (lambda_now * g_pos / (g_tok + 1e-12)) if (g_tok > 0 and g_pos > 0) else float("nan")
            rows.append((valid_i, g_tok, g_pos, g_pos / (g_tok + 1e-12), implied_r))

            del _x, _tix, _target
            torch.cuda.empty_cache()

    if not rows:
        print(f"[gradlog step={step}] no rows (lambda={lambda_now:.4g})")
        return

    arr = np.array(rows, dtype=np.float64)
    print("-"*88)
    print(f"[gradlog step={step}] lambda={lambda_now:.4g} | used={len(rows)}")
    print(f"  median valid={np.median(arr[:,0]):.0f} | median g_pos/g_tok={np.median(arr[:,3]):.3g} | median implied_r={np.nanmedian(arr[:,4]):.3g}")
    for (valid, gtok, gpos, ratio, r) in rows:
        print(f"    valid={int(valid):4d} g_tok={gtok:.6g} g_pos={gpos:.6g} gpos/gtok={ratio:.3g} implied_r={r:.3g}")
    print("-"*88)

































































































































































# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    compute_done = (step == stop_after_step)
    raw_data_done = (args.continue_to_data_matching and data_match_raw_target > 0 and raw_tokens_seen >= data_match_raw_target)
    clean_data_done = (clean_example_match_target > 0 and clean_examples_seen >= clean_example_match_target)
    data_done = raw_data_done or clean_data_done

    # If continuing to a data-matched target, stop there. Otherwise stop at compute horizon.
    last_step = data_done if (args.continue_to_data_matching or clean_example_match_target > 0) else compute_done

    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes, edit_model=True)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        if args.train_mode == "ces" and args.ces_eval_tokens > 0:
            ces_eval_steps = max(
                1,
                args.ces_eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size),
            )
            ces_metrics = evaluate_ces_joint_bpb(
                model, build_ces_val_loader(), ces_eval_steps, token_bytes
            )
            print0(
                f"Step {step:05d} | CES joint bpb: {ces_metrics['joint_bpb']:.6f} "
                f"| token bpb: {ces_metrics['token_bpb']:.6f} "
                f"| position bits/pointer: {ces_metrics['position_bits_per_pointer']:.4f} "
                f"| position normalized nll: {ces_metrics['position_normalized_nll']:.4f} "
                f"| rho: {ces_metrics['edit_mass_rho']:.6f}"
            )
            wandb_run.log({"step": step, **{f"val/ces_{k}": v for k, v in ces_metrics.items()}})
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        with disable_fp8(orig_model):
            for title, prepend, prmpts, extra, no_edits in [
                (
                    "Pure auto-regressive generation",
                    "<|bos|>",
                    [
                        "The capital of France is",
                        "The chemical symbol of gold is",
                        "If yesterday was Friday, then tomorrow will be",
                        "The opposite of hot is",
                        "The planets of the solar system are:",
                        "My favorite color is",
                        "If 5*x + 3 = 13, then x is",
                    ],
                    [],
                    True,
                ),
                (
                    "Insertion-deletion opened and edit conditioned samples",
                    INDL_BOS,
                    [
                        "The capital of is Paris",
                        "The chemical symbol of is Au",
                        "If yesterday was, then tomorrow will be Sunday",
                        "The opposite of is cold",
                        "The of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune",
                        "My favorite is blue",
                        "If 5*x + 3 = , then x is 2",
                    ],
                    [tokenizer.encode_special(EDIT_START)],
                    False,
                )
            ]:
                print0("\n---")
                print0(title)
                print0("---")
                for prompt in prmpts:
                    tokens = tokenizer(prompt, prepend=prepend)
                    with autocast_ctx:
                        sample, _, pointers = engine.generate_batch(
                            tokens + extra, num_samples=1, max_tokens=16, temperature=0, no_edits=no_edits
                        )
                    print0(tokenizer.decode(sample[0]))
                    for _smp, _ptr in zip(sample, pointers):
                        print0("\tstatus.value='RAW_EXTENDED_EDIT_SAMPLE'; complete=False")
                        print0(f"ptr: {pointers[0]}")
                        print0()
                print0("---")
            print0(f"---\nPrompts conditioning on a single insertion:\n---")
            engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
            prmpts_markers = [
                ("The capital of is Paris", " of"),
                ("The chemical symbol of is Au", " of"),
                ("If yesterday was, then tomorrow will be Sunday", " was"),
                ("The opposite of is cold", " of"),
                ("The of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune", "The"),
                ("My favorite is blue", " favorite"),
                ("If 5*x + 3 = , then x is 2", " ="),
            ]
            extra = [
                tokenizer.encode_special(EDIT_START),
                tokenizer.encode_special(INS_OPEN),
            ]
            doc_bos = tokenizer.encode_special("<|bos|>")
            for prompt, mark in prmpts_markers:
                tokens = [doc_bos] + tokenizer(prompt, prepend=INDL_BOS) + extra
                seperated_tokens = tokenizer.enc.decode_batch([[t] for t in tokens])
                [next_pos] = [i for i, t in enumerate(seperated_tokens) if t == mark]
                eval_tix = list(range(len(tokens)))
                with autocast_ctx:
                    sample, _, pointers = engine.generate_batch(
                        tokens, tix=eval_tix, num_samples=1, max_tokens=16, temperature=0
                    )
                    print0(f"target-pointer: {next_pos}, sampled: {pointers[0][len(tokens)]}")
                    print0(tokenizer.decode(sample[0]))
                    tokens += [tokenizer.encode_special(INS_LOC)]
                    sample, _, pointers = engine.generate_batch(
                        tokens, tix=eval_tix + [next_pos], num_samples=1, max_tokens=16, temperature=0
                    )
                    dec = tokenizer.enc.decode([sample[0][len(tokens)]])
                    print0(f"given position, next token prediction: {dec}")
            print0(f"---\nPrompts conditioning on a single deletion:\n---")
            print0("---")
            extra = [
                tokenizer.encode_special(EDIT_START),
                tokenizer.encode_special(DEL_OPEN),
            ]
            for prompt in [
                "blah The capital of France is Paris",
                "The chemical blah blah blah symbol of gold is Au",
                "If yesterday was Friday, blah then tomorrow will be Sunday",
                "The opposite of hot blah is cold",
                "The planets of the solar blah blah system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune",
                "My favorite color blah blah is blue",
                "If 5*x + 3 blah = 13, then x is 2",
            ]:
                marks = {" bl", "bl", "ah"}
                tokens = [doc_bos] + tokenizer(prompt, prepend=INDL_BOS) + extra
                seperated_tokens = tokenizer.enc.decode_batch([[t] for t in tokens])
                del_positions = [i for i, t in enumerate(seperated_tokens) if t in marks]
                del_start, del_end = min(del_positions), max(del_positions)
                eval_tix = list(range(len(tokens)))
                with autocast_ctx:
                    sample, _, pointers = engine.generate_batch(
                        tokens, tix=eval_tix, num_samples=1, max_tokens=16, temperature=0
                    )
                    pred_start = pointers[0][len(tokens)]
                    tokens += [tokenizer.encode_special(DEL_START)]
                    sample, _, pointers = engine.generate_batch(
                        tokens, tix=eval_tix + [del_start], num_samples=1, max_tokens=16, temperature=0
                    )
                    print0(f"target-del-pointer: ({del_start}, {del_end}), sampled: ({pred_start}, {pointers[0][len(tokens)]})")
        model.train()

    # once in a while: run arithmetic generalization/edit validity eval (only on master process)



















    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    # if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
    save_now = compute_done or last_step or (
        step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0
    )
    if save_now:
        if delete_resume_checkpoint_before_save:
            if is_ddp_initialized():
                dist.barrier()
            delete_checkpoint(resume_checkpoint_dir, args.resume_from_step, rank=ddp_rank)
            if is_ddp_initialized():
                dist.barrier()
            print0(
                f"Deleted resume checkpoint before writing its replacement: "
                f"{resume_dirname} step {args.resume_from_step}"
            )
            delete_resume_checkpoint_before_save = False
        checkpoint_reason = (
            "clean_example_matched" if clean_data_done else
            "raw_token_matched" if raw_data_done else
            ("token_matched" if stop_after_step == num_iterations else "milestone") if compute_done else
            "periodic"
        )
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict() if args.save_optimizer else None, # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "tokenizer": tokenizer_name,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "checkpoint_reason": checkpoint_reason,
                "initialization": initialization_metadata,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "raw_tokens_seen": raw_tokens_seen,
                    "clean_examples_seen": clean_examples_seen,
                    "edit_mass_seen": edit_mass_seen,
                    "valid_pointer_targets_seen": valid_pointer_targets_seen,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    loader_wait_time = 0.0
    accaux = {k: 0 for k in ["position_loss", "token_loss"]}
    step_edit_mass = 0
    step_raw_tokens = 0
    step_valid_pointer_targets = 0
    for k in list(target.perplexity_masks or {}) + ["position_correct"]:
        accaux[k] = (0, 0)
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            model_output = model(x, tix, targets=target)
            if isinstance(model_output, tuple):
                loss, aux = model_output
            else:
                loss = model_output
                zero = loss.detach() * 0.0
                aux = {
                    "position_loss": zero,
                    "token_loss": loss,
                    "position_correct": (
                        torch.zeros((), device=device, dtype=torch.long),
                        torch.zeros((), device=device, dtype=torch.long),
                    ),
                }
        train_loss = loss.detach() # for logging
        loss = (
            aux["position_loss"] * get_pos_weight(step) + aux["token_loss"]
        ) / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        aux["position_loss"] = aux["position_loss"].detach()
        aux["token_loss"] = aux["token_loss"].detach()
        for k, v in aux.items():
            if isinstance(v, tuple):
                val, count = accaux[k]
                accaux[k] = (val + v[0], count + v[1])
            else:
                accaux[k] += v / grad_accum_steps

        raw_step_local = int(dataloader_state_dict.get("raw_token_consumption", 0))
        raw_step_t = torch.tensor(raw_step_local, device=device, dtype=torch.long)
        if ddp:
            torch.distributed.all_reduce(raw_step_t, op=torch.distributed.ReduceOp.SUM)
        raw_tokens_seen += int(raw_step_t.item())
        step_raw_tokens += int(raw_step_t.item())

        edit_step_t = torch.tensor(int(dataloader_state_dict.get("edit_mass", 0)), device=device, dtype=torch.long)
        pointer_step_t = torch.tensor(int(dataloader_state_dict.get("valid_pointer_targets", 0)), device=device, dtype=torch.long)
        if ddp:
            torch.distributed.all_reduce(edit_step_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(pointer_step_t, op=torch.distributed.ReduceOp.SUM)
        step_edit_mass += int(edit_step_t.item())
        step_valid_pointer_targets += int(pointer_step_t.item())

        clean_step_local = int(dataloader_state_dict.get("raw_examples_consumption", 0))
        clean_step_t = torch.tensor(clean_step_local, device=device, dtype=torch.long)
        if ddp:
            torch.distributed.all_reduce(clean_step_t, op=torch.distributed.ReduceOp.SUM)
        clean_examples_seen += int(clean_step_t.item())

        loader_t0 = time.time()
        (x, tix, target), dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        loader_wait_time += time.time() - loader_t0
    # step the optimizer
    t_opt0 = time.time()
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    new_only_phase = adaptation_active and step < args.adapt_new_only_steps
    for group in optimizer.param_groups:
        role = group.get("adapt_role", "backbone")
        role_scale = 1.0
        if adaptation_active:
            if new_only_phase:
                role_scale = 1.0 if role in {"new", "vocab"} else 0.0
            elif role in {"backbone", "vocab"}:
                role_scale = args.adapt_backbone_lr_scale
        group["lr"] = group["initial_lr"] * lrm * role_scale
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = 0.0 if new_only_phase else muon_weight_decay
        elif new_only_phase and role == "vocab":
            group["weight_decay"] = 0.0
        else:
            group["weight_decay"] = group.get("initial_weight_decay", group.get("weight_decay", 0.0))
    if scaler is not None:
        scaler.unscale_(optimizer)
        apply_adaptation_gradient_policy(step)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        apply_adaptation_gradient_policy(step)
        optimizer.step()
    model.zero_grad(set_to_none=True)
    total_loss = loss.item()  # .item() is a CPU-GPU sync point
    aux = {k: (v[0].item(), v[1].item()) if isinstance(v, tuple) else v.item() for k, v in accaux.items()}
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    optim_time = t1 - t_opt0
    compute_time = dt - loader_wait_time - optim_time
    last_loader_wait = loader_wait_time
    edit_mass_seen += step_edit_mass
    valid_pointer_targets_seen += step_valid_pointer_targets
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    smooth_train_loss, debiased_smooth_loss = debias(smooth_train_loss, aux["token_loss"])
    smooth_pos_loss, debiased_pos_loss = debias(smooth_pos_loss, aux["position_loss"])
    ppls = {
        k: np.exp(v[0] / v[1])
        for k, v in aux.items()
        if isinstance(v, tuple) and v[1] > 0
    }
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    token_str = f"| seen: {raw_tokens_seen/1_000_000:.1f}/{data_match_raw_target/1_000_000:.1f}MT " if args.continue_to_data_matching else ""
    example_str = f"| examples: {clean_examples_seen:,}/{clean_example_match_target:,} " if clean_example_match_target > 0 else f"| examples: {clean_examples_seen:,} "
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | tloss: {debiased_smooth_loss:.6f} | ploss: {debiased_pos_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | compute: {compute_time * 1000:.2f}ms | loader: {loader_wait_time * 1000:.2f}ms | optim: {optim_time * 1000:.2f}ms {token_str}{example_str}| tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/raw_tokens_seen": raw_tokens_seen,
            "train/clean_examples_seen": clean_examples_seen,
            "train/edit_mass_seen": edit_mass_seen,
            "train/edit_mass_rho": edit_mass_seen / max(raw_tokens_seen, 1),
            "train/valid_pointer_targets_seen": valid_pointer_targets_seen,
            "train/valid_pointers_per_example": valid_pointer_targets_seen / max(clean_examples_seen, 1),
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/compute_dt": compute_time,
            "train/loader_dt": loader_wait_time,
            "train/optim_dt": optim_time,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    if args.grad_balance_every > 0 and args.train_mode == "ces" and step % args.grad_balance_every == 0:
        log_grad_balance(step, investigation_examples, max_examples=6)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report

get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()

#
