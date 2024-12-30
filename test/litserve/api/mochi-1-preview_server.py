"""
https://huggingface.co/genmo/mochi-1-preview

from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/mochi-1-preview', local_dir='/mnt/data/llch/mochi-1-preview')
"""

"""
python3 ./demos/gradio_ui.py --model_dir /mnt/data/llch/mochi-1-preview/ --cpu_offload
export no_proxy="localhost,36.213.66.106,127.0.0.1,112.48.199.202,112.48.199.7"
demo.launch(server_name="0.0.0.0", server_port=16845)
"""
