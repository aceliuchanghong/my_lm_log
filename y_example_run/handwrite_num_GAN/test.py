import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from y_example_run.handwrite_num_GAN.GAN import Generator


def test(generator_path, device, num_samples=16):
    # 加载生成器模型参数
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()  # 切换到评估模式

    # 生成随机噪声
    noise = torch.randn(num_samples, 100, device=device)

    # 生成假样本
    with torch.no_grad():  # 关闭梯度计算
        fake_images = generator(noise).cpu()

    # 展示生成的图像
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    grid = vutils.make_grid(fake_images, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

    return fake_images


if __name__ == "__main__":
    run_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(run_device)
    test('generator.pth', device)
