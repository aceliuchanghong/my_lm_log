import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def show_images(dataloader, num_images=16):
    # 获取一批数据
    data_iter = iter(dataloader)
    images, _ = next(data_iter)

    # 展示图像
    plt.figure(figsize=(8, 8))  # 设置图形的尺寸为8x8英寸
    plt.axis("off")
    plt.title("MNIST Images")

    # 创建网格并展示图片
    grid = vutils.make_grid(images[:num_images], padding=2, normalize=True)  # 使用padding=2在图片之间加上间距
    plt.imshow(grid.permute(1, 2, 0))  # grid.permute(1, 2, 0)是将图像张量的维度进行变换。将张量维度从(C, H, W)变为(H, W, C)
    plt.show()


if __name__ == '__main__':
    dataloader = DataLoader(dsets.MNIST('./data', transform=transform, download=True), batch_size=128, shuffle=True)
    show_images(dataloader)
