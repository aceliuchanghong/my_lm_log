# Gradio 入门指南

## 什么是Gradio？

Gradio 是一个快速创建演示界面的Python库

## 基本组件

### 输入组件
Gradio 提供了多种输入组件来接收用户输入：
- Textbox：文本输入框
- Slider：滑动条
- Dropdown：下拉菜单
- Image：图片上传
- Audio：音频上传
- Video：视频上传

### 简单示例
```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"
    interface = gr.Interface(
        fn=greet, inputs="text", outputs="text", title="Greeting App"
    )
    interface.launch()
```

### 输出组件
Gradio 同样提供了丰富的输出组件来展示结果：
- Textbox：文本输出
- Label：分类标签
- Image：图片展示
- Audio：音频播放
- Video：视频播放
- JSON：结构化数据展示

## 主要特点
- 快速创建：几行代码即可创建交互界面
- 多种组件：支持多种输入输出类型
- 易于分享：可生成公共链接分享给他人
- 可扩展性：支持自定义组件和布局
- 集成方便：可与主流机器学习框架无缝集成

## 安装方法
使用pip安装：
```bash
pip install gradio
```

## 下一步
- 了解如何创建更复杂的输入界面
- 学习如何处理多种输出类型
- 探索Gradio的高级功能
