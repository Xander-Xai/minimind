# 学习路线 M00-M17

固定顺序：

M00: [仓库与环境](modules/M00_REPOSITORY_ENVIRONMENT.md)
M01: [Tokenizer](modules/M01_TOKENIZER.md)
M02: [Transformer](modules/M02_TRANSFORMER.md)
M03: [Causal Language Modeling](modules/M03_CAUSAL_LM.md)
M04: [Pretrain](modules/M04_PRETRAIN.md)
M05: [SFT](modules/M05_SFT.md)
M06: [LoRA / QLoRA](modules/M06_LORA_QLORA.md)
M07: [Preference Data](modules/M07_PREFERENCE_DATA.md)
M08: [DPO From Scratch](modules/M08_DPO_FROM_SCRATCH.md)
M09: [DPO with TRL](modules/M09_DPO_WITH_TRL.md)
M10: [Reward Model](modules/M10_REWARD_MODEL.md)
M11: [Reinforcement Learning Basics](modules/M11_RL_BASICS.md)
M12: [PPO / RLHF](modules/M12_PPO_RLHF.md)
M13: [KL / Critic / GAE](modules/M13_KL_CRITIC_GAE.md)
M14: [GRPO / RLAIF](modules/M14_GRPO_RLAIF.md)
M15: [DeepSpeed / DDP / FSDP](modules/M15_DISTRIBUTED.md)
M16: [Full Pipeline Review](modules/M16_FULL_PIPELINE_REVIEW.md)
M17: [Interview Drills](modules/M17_INTERVIEW_DRILLS.md)

P0：Pretrain、SFT、Preference、DPO、Reward Model、PPO、KL。P1：LoRA、QLoRA、GRPO、RLAIF、DeepSpeed。P2：ORPO、IPO、KTO、CISPO、Agentic RL（仅概念卡）。

学习循环固定为 READ → TRACE → DEBUG → RUN → MODIFY → COMPARE → EXPLAIN → INTERVIEW。TRACE 必须沿 dataset → collator → input_ids → labels → model.forward → logits → loss → backward → optimizer.step。DPO 追踪 prompt/chosen/rejected → policy/reference → sequence log probability → log ratio → beta → logsigmoid → loss → backward。PPO 追踪 prompt → rollout → actor → response → reward → critic → advantage → old/new logprob → ratio → clip → KL → policy/value loss。

第一阶段只审计和初始化；不下载完整数据，不开展长训练。每个课程实验先用几十/几百 step 的小规模设置，并记录真实证据。

详细源码问答审计见 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)。
