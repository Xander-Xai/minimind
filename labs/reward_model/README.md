# Reward Model Mini Lab 设计（尚未执行）

目标：输入 `(prompt, response)`，输出 scalar reward。偏好样本包含 `prompt`, `chosen`, `rejected`。

模型对两种回答分别计算 `r_chosen = RM(prompt, chosen)` 与 `r_rejected = RM(prompt, rejected)`，使用 `loss = -log sigmoid(r_chosen - r_rejected)`。chosen reward 要高于 rejected，因为标注表达 chosen 更受偏好；scalar 汇总整段回答供排序和 PPO 使用；生成模型输出词表概率以继续生成，RM 输出单个分数而不生成文本。

PPO rollout 后，RM 对 response 打分形成轨迹奖励，再结合 KL、critic 和 advantage 更新 policy。Reward hacking 会在 RM 学到长度、格式等偏差时发生，policy 可能优化可利用特征而不是答案质量；需验证 pair、长度对照与人工抽查。

计划实验：合成 32-128 对短样本，小 batch、几十步，记录 pair accuracy、margin、train/validation loss、shape、显存和 runtime。实验代码尚未实现。本阶段不下载数据、不启动训练。

MiniMind 只有 `trainer/trainer_utils.py` 中 `LMForRewardModel` 的 inference/scoring helper；没有独立 preference-based `train_reward_model.py`。mini_qwen 没有 RM training/scoring pipeline。
