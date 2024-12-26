## Fluxgym - LoRA微调框架评测

### 框架特点

| 特性 | Fluxgym | 其他框架 |
|------|---------|----------|
| VRAM需求 | 12GB/16GB/20GB | 通常需要24GB+ |
| 易用性 | 简单Web UI | 命令行操作 |
| 功能完整性 | 支持Kohya全部功能 | 功能可能受限 |
| 模型支持 | 自定义扩展 | 固定模型 |
| 样本生成 | 自动生成训练样本 | 手动生成 |

#### 评价
```
1. Fluxgym最大的优势在于低显存需求，12GB显存即可运行，大大降低了硬件门槛

2. 基于Kohya脚本开发，功能完整且可扩展，同时提供了简洁的Web UI，降低了使用难度

3. 支持自动样本生成功能，可以直观观察训练过程中的模型变化，便于调试

4. 模型支持灵活，可以通过修改models.yaml文件添加自定义基础模型

5. 提供Docker部署方式，简化了环境配置流程
```

![界面截图](../../z_using_files/img/judge/fluxgym_ui.png)
![训练流程](../../z_using_files/img/judge/fluxgym_flow.gif)

### 项目地址:
- [GitHub](https://github.com/cocktailpeanut/fluxgym)

### 使用文档:
- [官方README](https://github.com/cocktailpeanut/fluxgym#readme)