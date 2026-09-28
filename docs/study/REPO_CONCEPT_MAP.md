# 双仓库概念映射

路径相对仓库根目录。`NOT_IMPLEMENTED_IN_REPO` 表示审计 checkout 中没有发现相应实现。mini_qwen 的 Trainer/TRL 行为特指 README 记录的历史版本，避免将最新 API/默认值倒推到该项目。

| 概念 | MiniMind 源码 | mini_qwen 源码 | 原理 | 框架实现 | 是否需要补实验 |
|---|---|---|---|---|---|
| Tokenizer | trainer/train_tokenizer.py | AutoTokenizer Qwen tokenizer files | 文本编码成模型词表 id | MiniMind 自训练 BPE；mini_qwen 加载既有 tokenizer | 两边做 token trace |
| Transformer | model/model_minimind.py:MiniMindModel/Block | Qwen Transformers AutoModelForCausalLM | attention 与 MLP 堆叠 | MiniMind 自写；mini_qwen 调 HF | shape trace |
| Causal LM | MiniMindForCausalLM.forward | AutoModelForCausalLM / Trainer loss | next-token prediction | 显式 shift CE vs HF loss | labels/logits trace |
| Pretrain | trainer/train_pretrain.py + PretrainDataset | mini_qwen_pt.py | 无标注 next-token 学习 | 自定义 Dataset vs Datasets map/Trainer | 微型语料 |
| Cross Entropy | model_minimind.py:256-257 | HF causal LM loss + DataCollatorForLanguageModeling | 目标 token NLL | 显式 F.cross_entropy vs HF | shift 与 ignore_index |
| SFT | train_full_sft.py + SFTDataset | mini_qwen_sft.py + SFTTrainer | 指令/对话监督 | 自定义 response mask vs TRL completion collator | mask trace |
| LoRA | model/model_lora.py; trainer/train_lora.py | NOT_IMPLEMENTED_IN_REPO | 低秩权重增量 | MiniMind 有 adapter 路线 | 目标层/rank lab |
| QLoRA | NOT_IMPLEMENTED_IN_REPO | NOT_IMPLEMENTED_IN_REPO | 量化基座上的 LoRA | 需另建 lab | 仅设计 |
| Preference Data | dataset/lm_dataset.py:DPODataset | mini_qwen_dpo.py:preprocess_dataset | 同 prompt 下成对偏好 | tensor pair vs 三列文本 | schema lab |
| chosen | DPODataset chosen 分支 | mini_qwen_dpo.py chosen 字段 | 偏好胜出回答 | pairwise loss 正样本 | 检查字段对齐 |
| rejected | DPODataset rejected 分支 | mini_qwen_dpo.py rejected 字段 | 偏好落败回答 | pairwise loss 对照项 | 交换方向实验 |
| Reward Model | trainer_utils.py:LMForRewardModel inference/scoring；无 train_reward_model.py | NOT_IMPLEMENTED_IN_REPO | (prompt,response)→scalar | 缺 preference RM training | labs/reward_model 设计 |
| RLHF | trainer/train_ppo.py actor/reference/critic/reward | NOT_IMPLEMENTED_IN_REPO | 偏好反馈驱动 policy 优化 | MiniMind PPO pipeline；RM training 缺 | toy reward flow |
| Policy | train_ppo.py actor_model | DPOTrainer policy model；无 RL loop | 生成 token 概率分布 | 自定义 actor vs TRL model | logprob trace |
| Reference Model | train_dpo.py / train_ppo.py frozen ref | DPOTrainer v0.11.4 reference handling | KL/log-ratio 基准策略 | 显式冻结 vs TRL：普通模型复制，PEFT 关闭 adapter | 确认冻结/独立性 |
| Critic | train_ppo.py:CriticModel | NOT_IMPLEMENTED_IN_REPO | 估计状态价值 V(s) | MiniMind value head | value shape lab |
| Value | CriticModel.forward | NOT_IMPLEMENTED_IN_REPO | 未来回报估计 | MiniMind token value path | return lab |
| Reward | train_ppo.py:calculate_rewards; train_grpo.py | NOT_IMPLEMENTED_IN_REPO | 标量/规则反馈信号 | MiniMind rules + optional RM scorer | 奖励偏差审计 |
| PPO | trainer/train_ppo.py | NOT_IMPLEMENTED_IN_REPO | clipped policy gradient 与 value objective | MiniMind 手写 actor/critic | ratio/clip toy |
| KL | train_ppo.py reference logps; train_grpo.py beta penalty | DPOTrainer 内部 DPO objective；无 RL KL loop | 限制偏离 reference | 显式 token KL/penalty vs TRL DPO | beta sensitivity |
| Advantage | train_ppo.py GAE; train_grpo.py group reward | NOT_IMPLEMENTED_IN_REPO | 相对 baseline 的动作价值 | GAE vs group baseline | 手算 |
| GAE | train_ppo.py reverse TD recursion | NOT_IMPLEMENTED_IN_REPO | 折中估计 advantage | gamma/lambda recursion | 三步递推 |
| DPO | trainer/train_dpo.py | mini_qwen_dpo.py + DPOTrainer | 离线 preference objective | 手写 vs TRL 封装 | logratio lab |
| DPO beta | train_dpo.py loss/CLI beta；CLI 默认 0.15 | mini_qwen_dpo.py 未传 beta；TRL 0.11.4 默认 0.1 | 控制 margin 尺度/偏离 reference 的约束 | 函数默认值与 CLI 值需区分；TRL pinned config | 改变 beta |
| GRPO | trainer/train_grpo.py | NOT_IMPLEMENTED_IN_REPO | 同 prompt group baseline，无 critic GAE | MiniMind 手写 | group reward lab |
| RLAIF | RLAIFDataset + heuristic/optional reward scoring | NOT_IMPLEMENTED_IN_REPO | AI/规则反馈支持对齐 | MiniMind 有部分数据/评分组件，不是完整 AI judge pipeline | 设计 evaluator |
| ORPO | NOT_IMPLEMENTED_IN_REPO | NOT_IMPLEMENTED_IN_REPO | odds-ratio preference objective | 未实现 | 仅概念卡 |
| DeepSpeed | NOT_IMPLEMENTED_IN_REPO | accelerate_config.yaml + run.sh | ZeRO 分片训练状态 | Accelerate + DeepSpeed ZeRO-2 | 配置审读 |
| DDP | trainer_utils.py + trainers DistributedDataParallel | Accelerate distributed launch config selects DeepSpeed | 复制模型、同步梯度 | MiniMind explicit DDP；mini_qwen Accelerate/DeepSpeed | rank/batch math |
| FSDP | NOT_IMPLEMENTED_IN_REPO | NOT_IMPLEMENTED_IN_REPO | 参数/梯度/优化器状态分片 | 未接入 | 概念卡 |
| IPO | NOT_IMPLEMENTED_IN_REPO | NOT_IMPLEMENTED_IN_REPO | implicit preference optimization | 未实现 | 仅概念卡 |
| KTO | NOT_IMPLEMENTED_IN_REPO | NOT_IMPLEMENTED_IN_REPO | prospect-theoretic preference optimization | 未实现 | 仅概念卡 |
| CISPO | NOT_IMPLEMENTED_IN_REPO | NOT_IMPLEMENTED_IN_REPO | clipped importance sampling preference optimization | 未实现 | 仅概念卡 |
| Agentic RL | trainer/train_agent.py 部分 agent rollout/reward loop | NOT_IMPLEMENTED_IN_REPO | 工具/多步交互策略优化 | MiniMind agent-specific route；不等于通用完整方案 | 仅概念卡 |
