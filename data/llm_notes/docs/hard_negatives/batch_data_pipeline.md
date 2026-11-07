# NLP 批处理数据管线

不同长度的文本组成 batch 时，Tokenizer 通常加入 padding token，并输出 attention_mask 字段供模型识别有效位置。数据管线还要统一截断长度、特殊 token、标签对齐和动态 padding 策略，避免无效填充造成额外计算。

这里的 attention_mask 是批处理接口中的张量字段，其形状和取值约定由具体模型实现决定。排查数据错误时应先打印 token、长度和 mask，再确认标签是否与原始样本对应，不应仅根据变量名称推断模型内部的注意力计算方式。
