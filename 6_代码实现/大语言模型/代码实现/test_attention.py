import torch
import torch.nn.functional as F


def attention(query, key, value, mask=None):
    """
     实现注意力机制
     :param query: 查询向量 [batch_size, seq_len, d_k]
     :param key: 键向量 [batch_size, seq_len, d_k]
     :param value: 值向量 [batch_size, seq_len, d_v]
     :param mask: 可选掩码 [batch_size, seq_len, seq_len]
     :return: 注意力输出和注意力权重


    - 在实际的Transformer模型中，Q、K、V通常**不是**相同的初始值
    - 它们是通过不同的线性变换从同一个输入向量得到的：
      Q = W_q * X
      K = W_k * X
      V = W_v * X
    - 其中W_q、W_k、W_v是不同的可学习参数矩阵
    """
    d_k = query.size(-1)

    # 1. 计算注意力分数 `matmul`矩阵乘法函数。用于计算两个张量的矩阵乘积
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=query.dtype)
    )

    # 2. 应用掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. 通过softmax得到注意力权重
    p_attn = F.softmax(scores, dim=-1)

    # 4. 加权求和得到输出
    return torch.matmul(p_attn, value), p_attn


# uv run 6_代码实现/大语言模型/代码实现/test_attention.py
batch_size = 2
seq_len = 3
d_k = 4
d_v = 5

query = torch.randn(batch_size, seq_len, d_k)
key = torch.randn(batch_size, seq_len, d_k)
value = torch.randn(batch_size, seq_len, d_v)

output, attention_weights = attention(query, key, value)
print("输出形状:", output.shape)
print("注意力权重形状:", attention_weights.shape)
