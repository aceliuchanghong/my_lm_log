# Hugging Face 模型下载手册

本手册将指导您如何使用 Hugging Face 的命令行工具 `huggingface-cli` 下载模型，并将其保存到本地目录。我们将以 `mistralai/Pixtral-12B-2409` 和 `nvidia/Cosmos-1.0-Guardrail` 两个模型为例进行说明。

## 前提条件

1. **Python 环境**：确保您已经安装了 Python 3.6 或更高版本。
2. **Hugging Face 账户**：您需要一个 Hugging Face 账户，并获取到您的 `hf_token`（Hugging Face 访问令牌）。

## 安装依赖

首先，您需要安装 `huggingface_hub` 库，它提供了与 Hugging Face Hub 交互的命令行工具。

```bash
pip install -U huggingface_hub
```

## 下载模型

### 1. 下载 `mistralai/Pixtral-12B-2409` 模型

使用以下命令将 `mistralai/Pixtral-12B-2409` 模型下载到本地的 `checkpoints/Pixtral-12B-2409` 目录中：

```bash
huggingface-cli download --resume-download mistralai/Pixtral-12B-2409 --local-dir checkpoints/Pixtral-12B-2409 --token hf_token
```

#### 参数说明：
- `--resume-download`：如果下载中断，可以继续从中断处恢复下载。
- `mistralai/Pixtral-12B-2409`：要下载的模型名称。
- `--local-dir checkpoints/Pixtral-12B-2409`：指定模型下载后保存的本地目录。
- `--token hf_token`：您的 Hugging Face 访问令牌。

### 2. 下载 `nvidia/Cosmos-1.0-Guardrail` 模型

使用以下命令将 `nvidia/Cosmos-1.0-Guardrail` 模型下载到本地的 `checkpoints/Cosmos-1.0-Guardrail` 目录中：

```bash
huggingface-cli download --resume-download nvidia/Cosmos-1.0-Guardrail --local-dir checkpoints/Cosmos-1.0-Guardrail --token hf_token
```

#### 参数说明：
- `--resume-download`：如果下载中断，可以继续从中断处恢复下载。
- `nvidia/Cosmos-1.0-Guardrail`：要下载的模型名称。
- `--local-dir checkpoints/Cosmos-1.0-Guardrail`：指定模型下载后保存的本地目录。
- `--token hf_token`：您的 Hugging Face 访问令牌。

## 注意事项

1. **网络连接**：确保您的网络连接稳定，特别是在下载大型模型时。
2. **存储空间**：检查本地存储空间是否足够，特别是对于大型模型。
3. **访问权限**：确保您有权限访问所需的模型。某些模型可能需要特定的权限或访问令牌。

## 常见问题

### 1. 下载中断怎么办？
使用 `--resume-download` 参数可以从中断处继续下载。

### 2. 如何获取 Hugging Face 访问令牌？
登录 Hugging Face 网站，进入您的账户设置，找到 "Access Tokens" 部分，生成一个新的令牌。

### 3. 下载速度慢怎么办？
可以尝试使用代理或更换网络环境来提升下载速度。

## 结论

通过本手册，您已经学会了如何使用 `huggingface-cli` 工具从 Hugging Face Hub 下载模型，并将其保存到本地目录。希望这对您的项目有所帮助！

如果您有任何问题或需要进一步的帮助，请参考 Hugging Face 的官方文档或社区支持。

---

**注意**：请将 `hf_token` 替换为您的实际 Hugging Face 访问令牌。