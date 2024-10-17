### 循环神经网络（RNN)

(input0, state0) -> LSTM RNN -> (output0, state1);

(input1, state1) -> LSTM RNN -> (output1, state2);

...

(inputN, stateN)-> LSTM RNN -> (outputN, stateN+1);

#### 1. 核心概念
循环神经网络（RNN，Recurrent Neural Network）是一种专门用于处理序列数据的神经网络。RNN通过隐藏状态（hidden state）来记忆之前的输入，使其能够捕捉到时间步之间的依赖关系。这使得RNN特别适合处理时间序列、自然语言处理等涉及顺序信息的问题。

- **输入序列**：RNN接受输入的序列数据，例如时间序列或句子。
- **隐藏状态**：每个时间步的输出不仅依赖于当前的输入，还依赖于前一个时间步的隐藏状态。
- **参数共享**：RNN在不同的时间步之间共享参数，这减少了模型的复杂度。

#### 2. 常见问题及解答

- **为什么RNN适合处理序列数据？**
  RNN具有“记忆”功能，通过隐藏状态的传递，可以捕捉到输入序列中的时序依赖关系，因此特别适合处理时间序列、文本等。

- **什么是梯度消失/爆炸问题？**
  在长序列中，由于反向传播算法的逐层链式更新，梯度会随着时间步长的增加而逐渐消失或爆炸，导致模型训练效果差。这是RNN的一个经典问题，解决方法包括使用LSTM或GRU等改进模型。

- **如何选择RNN、LSTM和GRU？**
  LSTM和GRU是RNN的改进版本，能够缓解梯度消失问题。LSTM有更复杂的结构和更多的参数控制记忆和遗忘过程，而GRU是LSTM的简化版本，计算速度更快，性能在某些任务上接近LSTM。

#### 3. 代码实现

下面是使用PyTorch实现一个简单的RNN分类模型的代码示例：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)  # 初始化隐藏状态
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 只取最后时间步的输出
        return out

# 示例：定义模型
input_size = 10  # 输入特征维度
hidden_size = 20  # 隐藏层神经元数量
output_size = 2   # 输出类别数量

model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### 4. 数学公式理解

RNN的核心公式可以用以下步骤描述：

- 每个时间步$t$的隐藏状态$h_t$通过当前输入$x_t$和上一个时间步的隐藏状态$h_{t-1}$来计算：
  
  $$
  h_t = \sigma(W_h h_{t-1} + W_x x_t + b_h)
  $$

  其中，$\sigma$是激活函数，$W_h$和$W_x$是权重矩阵，$b_h$是偏置。

- 输出$y_t$通过隐藏状态$h_t$计算得出：

  $$
  y_t = W_y h_t + b_y
  $$

RNN的难点在于其长期依赖的捕捉能力，长序列可能导致梯度消失或爆炸。因此，改进的RNN结构（如LSTM和GRU）引入了门机制来有效控制信息的流动。