# LLM-Learning-OS Integration

This repository is the **mechanism-first execution/lab repository** for the cross-repository LLM training Track.

## Control plane

Canonical orchestration lives in:

- [LLM-Learning-OS PR #4](https://github.com/Xander-Xai/LLM-Learning-OS/pull/4)
- [Dual-Repository LLM Training Track](https://github.com/Xander-Xai/LLM-Learning-OS/blob/feat/link-minimind-mini-qwen/docs/tracks/deep-engineering/llm-training/README.md)
- [Dual-Repository PT/SFT/DPO Project](https://github.com/Xander-Xai/LLM-Learning-OS/blob/feat/link-minimind-mini-qwen/docs/projects/llm-training-dual-repo/README.md)

## This repository owns

Use MiniMind to expose and manipulate the underlying mechanisms:

- tokenizer training;
- decoder-only Transformer internals;
- causal-LM shift/loss;
- explicit PT/SFT data and label paths;
- LoRA;
- DPO from scratch;
- PPO / KL / critic / GAE;
- GRPO / RLAIF-style training;
- rollout and DDP behavior.

## This repository does not own

Do not turn `docs/study/` into a second Canonical Knowledge base.

- durable concept definitions belong in `LLM-Learning-OS`;
- repository/version-specific traces stay here;
- experiments and debug evidence stay here;
- only validated conclusions are distilled back into the OS;
- unsupported gaps remain explicit gaps.

## Pairing rule

For shared stages, use MiniMind first to answer **how the mechanism works**, then compare `Xander-Xai/mini_qwen` to answer **how Transformers / TRL / Accelerate / DeepSpeed packages or automates it**.

The local M00-M17 roadmap remains the execution checklist; LLM-Learning-OS defines the cross-repository outcome, assessment and Stop Rule.
