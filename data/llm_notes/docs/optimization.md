# 深度学习训练基础

交叉熵用于衡量模型预测分布与真实标签之间的差异。单标签分类中，若真实类别为 y，则单样本损失为负对数概率 -log p_y。PyTorch 的 cross_entropy 直接接收 logits，内部会使用数值稳定的 log_softmax 与负对数似然，因此不应提前手动调用 Softmax。

反向传播利用链式法则从损失函数开始，沿计算图反向计算每个参数的梯度。标准训练循环依次执行 optimizer.zero_grad、前向传播、loss.backward 和 optimizer.step。PyTorch 默认累积梯度，如果不清零，多次 backward 的梯度会相加。

Adam 同时维护梯度一阶矩和二阶矩的指数移动平均。一阶矩类似动量，用于平滑更新方向；二阶矩反映近期梯度尺度，为不同参数调整有效步长。Adam 还会在训练初期进行偏差修正。AdamW 将权重衰减与梯度更新解耦，是 Transformer 训练中的常见选择。

Warmup 在训练初期逐渐增大学习率，可以减少参数尚未稳定时的大幅更新。Cosine decay 随训练推进平滑降低学习率。Gradient clipping 用于限制梯度范数，降低梯度爆炸风险；gradient accumulation 则通过累计多个小批次的梯度模拟更大的有效 batch。

残差连接的形式为 y=x+F(x)。反向传播时存在恒等路径，使梯度即使在 F 分支较小时仍能传播。残差连接并不能单独解决所有训练问题，通常需要与归一化、初始化和学习率策略共同使用。

LayerNorm 通常对每个 token 的隐藏特征维度独立归一化，不依赖同一 batch 中的其他样本。它适合 batch size 和序列长度可能变化的 NLP 模型；BatchNorm 则依赖批次统计量，并在训练和推理阶段维护不同的行为。
