# Source Audit: MiniMind 与 mini_qwen

静态阅读记录，2026-09-27。路径相对各自仓库根目录；行号按记录时 checkout。未运行训练、未下载数据。模型/框架内部行为以仓库版本源码为准。

## MiniMind 源码地图

### 架构与语言建模

- Tokenizer：`trainer/train_tokenizer.py:get_texts/train_tokenizer` 读取 JSONL 的 `text` 或 conversations 内容，以 `tokenizers.models.BPE`、ByteLevel pre-tokenizer 和 BpeTrainer 训练，保存 `tokenizer.json` 与 tokenizer config。脚本注释说仓库已带 tokenizer，重训会破坏模型/词表契约。默认输入样例路径仅用于学习。
- Transformer：`model/model_minimind.py`。`MiniMindModel.embed_tokens` 做 embedding；`MiniMindBlock` 串联 pre-norm attention 和 MLP；`Attention` 含 q/k/v/o projections、GQA heads、q/k RMSNorm、causal attention/SDPA 路径；`precompute_freqs_cis` 与 `apply_rotary_pos_emb` 实现 RoPE（可带 YaRN scaling）；`RMSNorm` 实现归一化；`FeedForward` 的 `down_proj(act(gate_proj(x))*up_proj(x))` 是 SwiGLU 形式；`MiniMindForCausalLM.lm_head` 输出词表 logits。模型 forward 接收 `[B,T]` ids，hidden `[B,T,H]`，logits `[B,T,V]`。
- Pretrain：`dataset/lm_dataset.py:PretrainDataset.__getitem__` 对 text tokenize、加 BOS/EOS、pad 到 max_length；labels 是 input_ids 副本，pad 改成 -100。`trainer/train_pretrain.py:train_epoch` 把 ids/labels 输入模型并优化其 loss。
- Cross entropy：`MiniMindForCausalLM.forward` 对 logits[:-1] 与 labels[1:] 做 shift，flatten 后调用 `F.cross_entropy(..., ignore_index=-100)`。
- SFT：`SFTDataset.create_chat_prompt` 经 tokenizer chat template 构造序列；`generate_labels` 默认全部 -100，只识别 `<|im_start|>assistant\n` 后的内容直到 `<|im_end|>\n`，将 response 与结束符位置设为目标 token。它与 PT 的差异是多轮消息格式与 assistant-only loss mask。
- LoRA：`model/model_lora.py` 定义 LoRA 相关模块；`trainer/train_lora.py` 是 LoRA 训练入口。QLoRA（量化基座上的 LoRA）未找到实现。

### DPO

- `DPODataset.__getitem__` 分别将 chosen/rejected 对话应用 chat template，编码/pad/truncate，生成 assistant response loss mask，再返回 `x_chosen/y_chosen/mask_chosen` 与 rejected 对应张量。
- `trainer/train_dpo.py:logits_to_log_probs` 用 log_softmax + gather 取每 token 目标 logp；`dpo_loss` 先按 mask 对序列求和，假设 batch 前半 chosen、后半 rejected，计算 policy log-ratio 与 reference log-ratio 的差，再 `-F.logsigmoid(beta * logits)`。
- `beta` 在损失缩放 log-ratio margin。函数默认 0.1，但 CLI `--beta` 默认 0.15。启动时用同一初始化来源创建 policy/ref，ref `eval()` 且 `requires_grad_(False)`。

### PPO / GRPO / rollout / distributed

- PPO actor 是 `actor_model`；`CriticModel` 继承 MiniMind causal LM backbone，用额外 scalar value head 给 token states 评分。
- Reward 在 `calculate_rewards` 中由规则项、重复惩罚和 `LMForRewardModel.get_score` 可选打分合成。后者是加载现成模型推理/评分，不是偏好 RM 训练。全仓没有独立 `train_reward_model.py`。
- rollout 由 `trainer/rollout_engine.py` 抽象：Torch engine 调 generation，返回 completion ids/text、per-token logps 和 mask；SGLang engine 可走外部推理服务并周期性更新 policy。PPO 把 rollout 的 per-token logp 保存为 old logp。
- PPO 在 `train_ppo.py` 用 ref model 取 response token logp 构造 KL 惩罚/监控；`ratio=exp(new_logp-old_logp)`，以 `clip_epsilon` 裁剪 surrogate policy objective；critic value loss另算。GAE 倒序递推 `delta_t=reward_t+gamma*V_(t+1)-V_t` 与 `A_t=delta_t+gamma*lambda*A_(t+1)`。
- GRPO 在 `train_grpo.py` 每 prompt 采样多条 completion，收集 reward 后按 group reshape，基于组均值/标准差标准化 advantage；无 PPO critic GAE。它仍含 reference KL 与 policy ratio/clipped objective。
- Distributed：各训练脚本和 `trainer/trainer_utils.py:init_distributed_mode` 使用 PyTorch DDP；未发现 DeepSpeed 或 FSDP 接入。

## mini_qwen 源码地图

### PT

- `mini_qwen_pt.py` 读取本地 Qwen2.5-0.5B-Instruct 的 `AutoConfig` 与 tokenizer；修改 hidden size、层数、注意力头配置后调用 `AutoModelForCausalLM.from_config`，因此权重随机初始化，不是加载 Qwen checkpoint。保留 Qwen config/tokenizer 是复用结构定义、词表和 token-id 对齐。
- 数据文本加结束符后 tokenize；batched map 把 token 列表展平 concat，按 `block_size` 切块，丢弃不能凑成完整 block 的尾部 token。`DataCollatorForLanguageModeling(mlm=False)` 构造 causal LM labels（padding 忽略）；模型的 causal-LM forward 提供 shift/loss；Trainer 调 model loss 完成 backward/optimizer step。
- `flash_attention_2` 通过 `attn_implementation` 请求 Transformers 模型使用 FlashAttention 2 kernel，以减少 attention 的中间内存/改善吞吐；要求匹配硬件、CUDA、PyTorch、flash-attn 版本，并非算法必需。

### SFT

- `formatting_prompts_func` 把 human/gpt 转成 Qwen im_start/im_end 对话字符串。`DataCollatorForCompletionOnlyLM` 以 assistant response template 定位 response 区间并屏蔽 prompt labels；SFTTrainer 包装 dataset 格式化、截断/packing、训练循环和日志，底层仍是 causal LM token loss。
- demo 与完整脚本逻辑同类，但 demo 使用更小的 block/max sequence 长度、少量数据等便于快速演示的设置；不能把 demo 配置误当成 README 描述的全量训练规模。

### DPO 与历史依赖

- `mini_qwen_dpo.py:preprocess_dataset` 将 prompt 包装为 user + assistant generation prefix，从 chosen/rejected 的对话第二项提取 assistant content 并补 im_end，输出三列字符串。
- `DPOTrainer` 封装 tokenization/padding、policy/reference log probabilities、DPO objective、训练循环与分布式集成。脚本未显式传 beta；README 指定 TRL 0.11.4，`DPOConfig.beta` 默认 0.1。0.11.4 的 DPOTrainer 对非-PEFT 且未给 ref_model 时复制 reference model；PEFT 情况复用关闭 adapters 的 base model 作为 reference。不能把这个默认推断成所有 TRL 版本的行为。
- README 明确写历史 `trl==0.11.4`、`transformers==4.45.0` 和 flash-attn；根目录没有锁定 requirements/environment 文件。未经独立兼容性审计不要升级。

### 分布式与缺项

- `accelerate_config.yaml` 选择 `distributed_type: DEEPSPEED`、ZeRO stage 2、bf16、`num_processes: 6`、gradient accumulation 16；`run.sh` 用 accelerate launch。该配置对应作者训练环境，不能在本机 RTX 5060 Ti 单卡上原样照跑。
- mini_qwen 未发现自定义 tokenizer 训练、LoRA/QLoRA、RM scoring/training、PPO、GRPO、RLAIF 或 FSDP 实现。源码主要覆盖 PT、SFT、DPO 与 Accelerate/DeepSpeed 示例。

## 必须保持的证据边界

README 描述的上游训练规模/硬件是仓库作者报告，不是本机复现。第一阶段只完成静态源码与配置审计。MiniMind 具备 reward inference/scoring，但 preference-based Reward Model training requires an additional learning lab；不要声称两个仓库已经完整实现 RM training。
