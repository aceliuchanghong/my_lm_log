"""
cd no_git_oic
git clone https://github.com/fishaudio/fish-speech.git
conda create -n fish-speech python=3.10
source activate fish-speech
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
apt install libsox-dev ffmpeg
apt install build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0
cd fish-speech
pip3 install -e .[stable]
"""

"""
export no_proxy="localhost,127.0.0.1"
python -m tools.api_server \
    --listen 0.0.0.0:8120 \
    --llama-checkpoint-path "/mnt/data/llch/fish-speech1.5" \
    --decoder-checkpoint-path "/mnt/data/llch/fish-speech1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq

vi /mnt/data/llch/my_lm_log/no_git_oic/fish-speech/tools/api_client.py
python -m tools.api_client \
    --url "http://0.0.0.0:8120/v1/tts" \
    --text "要输入的文本,怎么样跑步呢?" \
    --reference_audio "/mnt/data/llch/my_lm_log/no_git_oic/ylhtest.wav" \
    --reference_text "请介绍一下火炬电子" \
    --streaming True

vi /mnt/data/llch/my_lm_log/no_git_oic/fish-speech/tools/run_webui.py
python -m tools.run_webui \
    --llama-checkpoint-path "/mnt/data/llch/fish-speech1.5" \
    --decoder-checkpoint-path "/mnt/data/llch/fish-speech1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
    --compile
"""
