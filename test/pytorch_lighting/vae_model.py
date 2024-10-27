import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
import os
from dotenv import load_dotenv
import logging
from PIL import Image
import time
from functools import partial

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def spectral_norm(module):
    """init & apply spectral norm"""
    nn.init.xavier_uniform_(module.weight, 2**0.5)
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.data.zero_()

    return nn.utils.spectral_norm(module)


def activ_dispatch(activ):
    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
    }[activ.lower()]


def norm_dispatch(norm):
    # 这些归一化层通常需要一个参数,即输入特征图的通道数 num_features,用于初始化内部参数
    return {
        "none": nn.Identity,
        "in": partial(nn.InstanceNorm2d, affine=False),
        "bn": nn.BatchNorm2d,
    }[norm.lower()]


def w_norm_dispatch(w_norm):
    # NOTE Unlike other dispatcher, w_norm is function, not class.
    return {
        "none": lambda x: x,
        "spectral": spectral_norm,
    }[w_norm.lower()]


def pad_dispatch(pad_type):
    return {
        "zero": nn.ZeroPad2d,
        "replicate": nn.ReplicationPad2d,
        "reflect": nn.ReflectionPad2d,
    }[pad_type.lower()]


class ConvBlock(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        stride=1,
        padding=1,
        norm="none",
        activ="relu",
        bias=True,
        upsample=False,
        downsample=False,
        w_norm="none",
        pad_type="zero",
        dropout=0.1,
    ):
        super().__init__()
        if kernel_size == 1:
            assert padding == 0
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.dropout = nn.Dropout2d(p=dropout)

        activ = activ_dispatch(activ)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)

        self.activ = activ()
        self.pad = pad(padding)
        self.norm = norm(self.C_in)
        self.conv = w_norm(
            nn.Conv2d(self.C_in, self.C_out, kernel_size, stride, bias=bias)
        )

    def forward(self, x):
        """
        下采样(Downsampling):
        是指将特征图的空间分辨率降低。例如,2倍的平均池化(F.avg_pool2d(x, kernel_size=2))
        会将特征图的高和宽缩小为原来的一半,减少数据量的同时保留主要特征。常用于卷积神经网络(CNN)的池化层,以降低计算量和提取高层次特征。

        上采样(Upsampling):
        用于将特征图的空间分辨率提高。例如,F.interpolate(x, scale_factor=2, mode="nearest")
        会将特征图的高和宽扩大2倍。上采样通常在生成网络或分割网络中使用,帮助恢复图像的原始分辨率,使输出更加细致。
        """
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.dropout(x)
        x = self.pad(x)
        x = self.conv(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size=3,
        padding=1,
        upsample=False,
        downsample=False,
        norm="none",
        w_norm="none",
        activ="relu",
        pad_type="zero",
        dropout=0.1,
        scale_var=False,
    ):
        assert not (upsample and downsample)
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = ConvBlock(
            C_in,
            C_out,
            kernel_size,
            1,
            padding,
            norm,
            activ,
            upsample=upsample,
            w_norm=w_norm,
            pad_type=pad_type,
            dropout=dropout,
        )
        self.conv2 = ConvBlock(
            C_out,
            C_out,
            kernel_size,
            1,
            padding,
            norm,
            activ,
            w_norm=w_norm,
            pad_type=pad_type,
            dropout=dropout,
        )
        w_norm = w_norm_dispatch(w_norm)
        # 跳跃连接
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

    def forward(self, x):
        """
        normal: pre-activ + convs + skip-con
        upsample: pre-activ + upsample + convs + skip-con
        downsample: pre-activ + convs + downsample + skip-con
        => pre-activ + (upsample) + convs + (downsample) + skip-con
        """
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        # skip-con
        if hasattr(self, "skip"):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        out = out + x
        if self.scale_var:
            out = out / np.sqrt(2)
        return out


class ContentEncoder(nn.Module):
    def __init__(self, layers, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.sigmoid = sigmoid

    def forward(self, x):
        out = self.net(x)
        if self.sigmoid:
            out = nn.Sigmoid()(out)
        return out


class ContentDecoder(nn.Module):
    def __init__(self, layers, skips=None, out="sigmoid"):
        super().__init__()
        """
        nn.Sequential(*layers): 将传入的层按照顺序“组合”成一个模型，输入数据传入后会依次经过所有层。这样可以简化代码结构
        nn.ModuleList(layers): 仅将传入的层放入一个“列表”中，并不会自动执行前向传递forward
        """
        self.layers = nn.ModuleList(layers)
        if out == "sigmoid":
            self.out = nn.Sigmoid()
        elif out == "tanh":
            self.out = nn.Tanh()
        else:
            raise ValueError(out)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return self.out(x)


def content_enc_builder(
    C_in, C, C_out, norm="none", activ="relu", pad_type="reflect", content_sigmoid=False
):
    # content_enc_builder(1, 32, 256)
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm="in"),
        ConvBlk(C * 1, C * 2, 3, 2, 1),
        ConvBlk(C * 2, C * 4, 3, 2, 1),
        ConvBlk(C * 4, C * 8, 3, 2, 1),
        ConvBlk(C * 8, C_out, 3, 1, 1),
    ]

    return ContentEncoder(layers, content_sigmoid)


def content_dec_builder(C, C_out, norm="none", activ="relu", out="sigmoid"):

    ConvBlk = partial(ConvBlock, norm=norm, activ=activ)
    ResBlk = partial(ResBlock, norm=norm, activ=activ)

    layers = [
        ResBlk(C * 8, C * 8, 3, 1),
        ResBlk(C * 8, C * 8, 3, 1),
        ResBlk(C * 8, C * 8, 3, 1),
        ConvBlk(C * 8, C * 4, 3, 1, 1, upsample=True),
        ConvBlk(C * 4, C * 2, 3, 1, 1, upsample=True),
        ConvBlk(C * 2, C * 1, 3, 1, 1, upsample=True),
        ConvBlk(C * 1, C_out, 3, 1, 1),
    ]

    return ContentDecoder(layers, out=out)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )

    def forward(self, inputs):
        # 传入的是图片经过encoder后的feature maps
        # convert inputs from BCHW
        input_shape = inputs.shape

        # Flatten input ->[BC HW]
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # 得到编号
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay=0.999, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
