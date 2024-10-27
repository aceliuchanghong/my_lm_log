import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import lightning as L
import os
from dotenv import load_dotenv
import logging
from PIL import Image
import time
import sys
from torch import optim

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from test.pytorch_lighting.vae_model import (
    content_enc_builder,
    content_dec_builder,
    VectorQuantizerEMA,
    VectorQuantizer,
)

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def collate_fn(batch):
    # batch 是一个列表，每个元素是一个元组 (image_name, img)
    images = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    # logger.info(f"{images.shape}")  # [B,C,H,W]
    return images


class VAE_DATASET(data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.pic_path = root
        self.transform = transform
        self.images = self.read_file(self.pic_path)

    def read_file(self, pic_path):
        files_list = os.listdir(pic_path)
        file_path_list = [os.path.join(pic_path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def __getitem__(self, index):
        img = self.images[index]
        image_name = img[-5:-4]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return image_name, img  # 图片名称和[C,H,W]

    def __len__(self):
        return len(self.images)


class LitVAE(L.LightningModule):
    """
    VAE的架构:一个编码器,一个解码器,外加中间一个嵌入层
    损失函数为图像的重建误差与编码器输出与其对应嵌入之间的误差
    """

    def __init__(
        self, num_embeddings=100, embedding_dim=256, commitment_cost=0.25, decay=0.0
    ):
        super().__init__()
        self.encoder = content_enc_builder(1, 32, 256)
        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self.vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self.decoder = content_dec_builder(32, 1)

    def forward(self, x):
        valid_originals = x
        vq_output_eval = self.encoder(valid_originals)

        _, valid_quantize, _, _ = self.vq_vae(vq_output_eval)

        valid_reconstructions = self.decoder(valid_quantize)

        return valid_reconstructions

    def training_step(self, batch, batch_idx):

        data = batch
        data = data - 0.5  # 归一化到[-0.5, 0.5]
        x = data
        train_data_variance = torch.var(data)

        z = self.encoder(x)
        vq_loss, quantized, perplexity, _ = self.vq_vae(z)
        data_recon = self.decoder(quantized)

        recon_error = F.mse_loss(data_recon, data) / train_data_variance
        loss = recon_error.mean() + vq_loss.mean()

        # logger.info(f"train_loss:{loss}")
        # logger.info(f"recon_error:{recon_error.mean()}")
        # logger.info(f"perplexity:{perplexity.mean()}")
        # logger.info(f"vq_loss:{vq_loss.mean()}")

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


if __name__ == "__main__":
    # python test/pytorch_lighting/test_vae.py
    new_root = r"C:\Users\lawrence\PycharmProjects\VQ-Font\z_using_files\f2p_imgs\SourceHanSansCN-Medium_val"
    pic_size = 128
    batch_size = 196  # 本机3060
    num_workers = 0
    tensorize_transform = tv.transforms.Compose(
        [tv.transforms.Resize((pic_size, pic_size)), tv.transforms.ToTensor()]
    )

    start_time = time.time()
    new_dataset = VAE_DATASET(new_root, tensorize_transform)
    new_dataloader = data.DataLoader(
        new_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # num_workers=0时，数据加载是在主进程中进行的 >0时，数据加载会在多个子进程中并行进行
        collate_fn=collate_fn,  # 指定了如何将单个样本组合成一个批次（batch）
        drop_last=True,  # 是否丢弃最后一个不完整的批次
        pin_memory=True,  # 是否将数据加载到CUDA固定内存中 加速CPU到GPU的数据传输
    )
    # data = next(iter(new_dataloader))
    # logger.info(f"{data.shape}") # [B,C,H,W]
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"数据加载耗时: {elapsed_time:.2f}秒")

    start_time = time.time()
    model = LitVAE(
        num_embeddings=100, embedding_dim=256, commitment_cost=0.25, decay=0.999
    )

    # from lightning.pytorch.callbacks import ModelCheckpoint
    # checkpoint_callback = ModelCheckpoint(dirpath="my/path/", save_top_k=2, monitor="val_loss")

    trainer = L.Trainer(max_epochs=500)
    trainer.fit(model, new_dataloader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"训练耗时: {elapsed_time:.2f}秒")
