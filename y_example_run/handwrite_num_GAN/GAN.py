import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# 创建数据集
transform = transforms.Compose([
    # 将图像从PIL格式或numpy数组转换为PyTorch的张量格式，并且将像素值从[0, 255]缩放到[0, 1]。
    transforms.ToTensor(),
    # 对图像的每个通道进行标准化。这里使用的均值和标准差都是0.5
    # 标准化后像素值 = (原来的像素值-0.5)/0.5
    transforms.Normalize((0.5,), (0.5,))
])


# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.Encoder = nn.Sequential(
            # 输入维度100表示随机噪声向量的大小
            # 生成器的目标是将这个低维的随机噪声空间映射到高维的图像空间。逐步通过神经网络将100维的向量转换成28*28的图像。
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            # 将输出限制在[-1, 1]之间，这与图像数据预处理的标准化（均值为0.5，标准差为0.5）步骤匹配，使生成的图像符合训练图片的标准化范围。
            nn.Tanh()
        )

    def forward(self, input):
        return self.Encoder(input).reshape(-1, 1, 28, 28)


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            # 将判别器的输出转换到(0, 1)的概率范围内，便于处理与真实和生成数据相关的二分类任务。
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.Encoder(input.view(-1, 28 * 28))


# 封装训练过程为train函数
def train(generator, discriminator, dataloader, criterion, optimizerG, optimizerD, device, total_epochs=20):
    for epoch in range(total_epochs):
        for i, data in enumerate(dataloader):
            # 更新判别器
            real_data, _ = data
            real_data = real_data.to(device)  # [B,C,H,W]
            batch_size = real_data.size(0)
            labels_real = torch.ones(batch_size, 1).to(device)
            labels_fake = torch.zeros(batch_size, 1).to(device)  # [B,1]

            optimizerD.zero_grad()
            output_real = discriminator(real_data)
            loss_real = criterion(output_real, labels_real)
            loss_real.backward()

            noise = torch.randn(batch_size, 100, device=device)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data.detach())
            loss_fake = criterion(output_fake, labels_fake)
            loss_fake.backward()
            optimizerD.step()

            # 更新生成器
            optimizerG.zero_grad()
            output_fake_for_G = discriminator(fake_data)
            loss_G = criterion(output_fake_for_G, labels_real)
            loss_G.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f'Epoch [{epoch + 1}/{total_epochs}],\
                     Step [{i + 1}/{len(dataloader)}], \
                     D Loss: {loss_real + loss_fake:.4f}, \
                     G Loss: {loss_G:.4f}')

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')


def main(total_epochs, device):
    # 初始化网络、损失函数和优化器
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=0.0002)
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002)

    # 创建数据集和数据加载器
    dataloader = DataLoader(dsets.MNIST('./data', transform=transform, download=True), batch_size=2048, shuffle=True)

    # 调用训练函数
    train(generator, discriminator, dataloader, criterion, optimizerG, optimizerD, device, total_epochs=total_epochs)


if __name__ == "__main__":
    run_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(run_device)
    main(total_epochs=200, device=device)
