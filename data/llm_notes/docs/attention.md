# Attention 与 Transformer

Self-Attention 让序列中的每个位置根据内容，从同一序列的所有位置聚合信息。给定输入 X，模型通过三个可学习线性映射得到 Query、Key 和 Value。Query 表示当前位置希望寻找什么信息，Key 表示每个位置可用于匹配的特征，Value 是匹配后真正被聚合的内容。

缩放点积注意力写作 softmax(QK^T / sqrt(d_k))V。若 Q 和 K 的分量近似独立且方差为 1，点积的方差会随维度 d_k 增长。维度较大时，点积绝对值容易变大，使 Softmax 进入饱和区域并产生很小的梯度。除以 sqrt(d_k) 可以把点积的方差控制在较稳定的范围。

Multi-Head Attention 把模型维度拆分为多个子空间，每个头使用独立的 Q、K、V 投影学习不同匹配模式。各头输出拼接后再经过输出投影。若模型维度为 d_model，头数为 h，通常每头维度 d_h=d_model/h，因此 d_model 必须能被头数整除。

Padding Mask 用于屏蔽补齐 token；Causal Mask 用于禁止当前位置看到未来 token。Mask 通常在 Softmax 前把不允许关注的位置设为负无穷，使其归一化后的权重接近零。

标准 Attention 的注意力矩阵大小为序列长度的平方，因此时间和显存包含 O(n^2) 项。它的优势是序列位置间交互路径短，并且训练阶段可以进行高度并行的矩阵计算。

Transformer 子层通常配合残差连接和归一化。Pre-LN 结构为 x + Sublayer(LN(x))，Post-LN 结构为 LN(x + Sublayer(x))。Pre-LN 的残差主路径更直接，通常更容易稳定训练深层网络。
