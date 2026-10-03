# Model names and training lineage

Models use `<method>-d<depth>-<stage>-seed<seed>[-<variant>]`. Training steps remain separate: a checkpoint is identified by its model ID and step. Existing artifact directories and `model_*.pt` filenames are stable download paths.

Stages are `pretrained`, `sft-coedit`, `sft-balanced`, `rl-broad`, `rl-gec`, and `sft-gec-matched`. Seeds identify model initialization for pretraining and the training stage seed for post-training; metadata records the seed role explicitly.

`docs/model-lineage.json` records complete training chains. Each metadata file also records its chain and immediate parent. Final GEC RL models follow pretrained → balanced SFT (800 updates) → broad RL (100) → GEC RL (200). All three RL seeds at a given depth share the seed-42 balanced SFT checkpoint. Intermediate SFT and broad-RL weights are not included.

The d12 RL family starts from `ces-d12-pretrained-seed42-posttraining-base` at step 74800, in `checkpoints/rl-posttraining/ces/d12/pretrained/`. This is a separate pretraining run from the methods-grid `ces-d12-pretrained-seed42` at step 26880. Its batch size is 16384 rather than 32768, and its corruption/stream settings and position-loss weight differ. The d18 and d24 RL families start from their released methods-grid pretrained models at steps 39600 and 74800. The d12 balanced-SFT recipe uses the matching post-training base.

Adapted-CES is one experiment with clean-AR pretraining followed by CES adaptation, optionally followed by CoEdIT SFT. Only the adapted and CoEdIT-finetuned checkpoints are distributed; its internal clean-pretraining stage is recorded as part of the recipe, not offered as another released model. Training adaptation separately requires the corresponding clean stage in `CES_INITIAL_CHECKPOINT`; the standalone clean-AR comparison models are a different experiment.

The two pointer ablations use `pointer-role-specific` and `pointer-shared` suffixes. Full names distinguish evaluations across methods and depths; short labels remain suitable within a clearly labelled table.
