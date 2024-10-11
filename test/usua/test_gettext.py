import gettext

# msgfmt -o test.mo  speed_benchmark.po
# 加载翻译
lang = gettext.translation('test', localedir='../z_using_files', languages=['zh_CN'])
# lang.install()  # 安装后 _() 函数会自动绑定为 gettext.gettext
_ = lang.gettext  # 显式定义 _ 函数

print(_("The environment of the evaluation with huggingface transformers is:"))
print(_("[Setting 3]=(gpu_memory_utilization=1.0 max_model_len=8192 enforce_eager=True)"))
print(_("CUDA 11.8"))
