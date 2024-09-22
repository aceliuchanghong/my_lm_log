## 支持向量机（SVM, Support Vector Machine）

是一种用于分类和回归分析的监督学习模型。它的核心思想是找到一个最优的决策边界，将数据点分成不同的类。

### 核心概念：
决策边界（超平面）：
SVM的目的是找到一个最优的决策边界（即一个超平面），可以将数据分为不同的类。这个边界应该尽可能使不同类别的样本之间的间隔（即“间隔最大化”）最大。

支持向量：
在SVM中，最重要的点是那些靠近决策边界的点，这些点被称为“支持向量”。支持向量决定了决策边界的位置。其他更远离边界的点对模型的影响较小。

间隔最大化：
SVM的目标是找到一个边界，使得不同类别的样本与边界的距离（间隔）最大化。这样做是为了提高模型的泛化能力，避免过拟合。

线性不可分和核技巧：
如果数据在原空间中无法被线性分割，SVM可以通过核函数（如多项式核、径向基函数核等）将数据映射到一个更高维的空间，在这个高维空间里找到一个线性可分的超平面。核技巧帮助SVM处理非线性分类问题。

### 公式解释：
SVM的数学表达式如下：

- 对于二分类问题，假设数据点为 $(x_i, y_i)$，其中 $(x_i)$ 是特征向量，$y_i \in \{-1, 1\}$ 是标签，SVM试图找到一个超平面：$w \cdot x + b = 0$，其中 $w$ 是权重向量，$b$ 是偏置。
- 目标是最大化决策边界两侧到支持向量的间隔，同时满足：对于所有 $i$，有 $y_i (w \cdot x_i + b) \geq 1$。

SVM通过优化这个问题，找到一个最优的 $w$ 和 $b$。

### 应用：
SVM广泛应用于图像分类、文本分类、生物信息学等领域，尤其是在数据规模不大但特征维度较高的情况下，SVM表现很好。

### 代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 数据准备
# 使用torchvision来加载MNIST数据集，并应用标准的图像预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. 定义SVM模型
class SVM(nn.Module):
    def __init__(self, input_size):
        super(SVM, self).__init__()
        # 由于每张图片是28x28像素，所以输入特征数是28*28 = 784
        self.linear = nn.Linear(input_size, 10)  # 10个类别，对应0到9的数字

    def forward(self, x):
        # 将输入展平成一维向量
        x = x.view(x.size(0), -1)
        return self.linear(x)

# 3. 定义损失函数和优化器
# Hinge Loss作为SVM的损失函数
def hinge_loss(output, target):
    target_one_hot = torch.zeros_like(output).scatter_(1, target.view(-1, 1), 1)
    return torch.mean(torch.clamp(1 - output * target_one_hot, min=0))

model = SVM(input_size=28*28)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = hinge_loss(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# 5. 在测试集上评估模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

"""
数据预处理：使用torchvision.datasets.MNIST来加载MNIST手写数字数据集。为了让数据适合SVM处理，应用了基本的归一化操作。

SVM模型：模型的输入为28×28=784个像素值，将图像展平为一维向量输入到线性层。输出层包含10个节点，每个节点对应数字0-9的分类。

Hinge Loss：由于SVM模型使用的是Hinge Loss，通过torch.clamp函数计算每个样本的损失，确保最小化分类边界的损失。

训练：使用随机梯度下降（SGD）来更新模型的参数，并输出每个epoch的损失值。

测试：训练完成后，在测试集上评估模型的准确性。
"""
```
