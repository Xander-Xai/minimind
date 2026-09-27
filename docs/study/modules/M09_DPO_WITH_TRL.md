# M09 DPO with TRL

## 1. 本课目标

Understand TRL DPOTrainer contract and historical API constraints.

## 2. 专业解释

Understand TRL DPOTrainer contract and historical API constraints. 核心公式/计算关系：`DPO pairwise log-ratio objective; beta default=0.1 in pinned TRL 0.11.4`。

## 3. 大白话解释

把训练过程当作一条加工线：每一站都核对输入、变换和输出；知道文件名不等于验证了实际运行。

## 4. 数学公式

`DPO pairwise log-ratio objective; beta default=0.1 in pinned TRL 0.11.4`

## 5. 数据流

```text
dataset → collator/preprocess → input_ids → labels → model.forward → logits → loss → backward → optimizer.step
本课特别关注：Understand TRL DPOTrainer contract and historical API constraints.
```

## 6. MiniMind 对应源码

mini_qwen_dpo.py: preprocess_dataset, DPOConfig, DPOTrainer

## 7. mini_qwen 对应源码

 README pins trl==0.11.4, transformers==4.45.0. In TRL v0.11.4, DPOConfig beta defaults to 0.1; DPOTrainer creates a reference copy for a non-PEFT model when none is supplied, while PEFT uses the base model with adapters disabled.

## 8. 两个实现为什么不同

MiniMind 在若干阶段提供自定义模型/训练循环；mini_qwen 更依赖 Transformers、TRL 和 Accelerate。对未实现项标注 NOT_IMPLEMENTED_IN_REPO；框架封装的内部默认值依赖对应版本。

## 9. Debug 路径

断点从 Dataset/preprocess 开始，经 collator、forward、loss 到 backward/optimizer.step。观察：input_ids/labels `[B,T]`、logits `[B,T,V]`；依课程再观察 token logp、reward、value 或 adapter 参数。核对 mask、dtype、device 和有效 token 数。

## 10. Mini Lab

Read code/API only; do not upgrade dependencies or load a model. 实验仅使用本地合成小样本；启动前写出预期。不得把理论推演记为实际运行。

## 11. 必改实验

改变本课一个核心参数或输入（如 block_size、mask、beta、chosen/rejected、clip epsilon、lambda），先预测，再用最小样本对照。

## 12. 面试问题

- 基础：DPO with TRL解决什么问题？
- 进阶：公式每一项对应哪个源码变量和 tensor？
- 追问：两个仓库为何采用不同实现？有哪些版本或未实现边界？

## 13. 脱稿验收

不看资料画出数据流、写核心公式、指出两仓源码路径、给出关键 tensor shape，并用大白话解释。只有理论、读码、真实 Debug、真实 Lab、Explain 和 Interview 全部通过才可 PASS。
