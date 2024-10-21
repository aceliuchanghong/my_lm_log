from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
import argparse


def generate_tts_file(
    session_id, input_text, reference_audio_path, reference_text, output_path
):
    session = Session(session_id)

    with open(reference_audio_path, "rb") as audio_file:
        with open(output_path, "wb") as f:
            for chunk in session.tts(
                TTSRequest(
                    text=input_text,
                    references=[
                        ReferenceAudio(
                            audio=audio_file.read(),
                            text=reference_text,
                        )
                    ],
                )
            ):
                f.write(chunk)


if __name__ == "__main__":

    """    
    python test/litserve/api/fish_speech1.4_server.py \
        --input_text "打开文档提取网站,上传pdf合同文件,设置好基本参数,点击开始提取,然后等待一下,查看提取结果" \
        --reference_audio_path "z_using_files/mp3/登陆系统,进入层数和K值预测界面.wav" \
        --reference_text "登陆系统,进入层数和K值预测界面" \
        --output_path "no_git_oic/ADET.mp3"
    """
    parser = argparse.ArgumentParser(
        description="Process TTS file generation arguments."
    )
    parser.add_argument(
        "--key",
        default="233caab7218248c8b97d145eac26e905",
        help="API key for TTS service",
    )
    parser.add_argument(
        "--input_text", required=True, help="Text to be converted to speech"
    )
    parser.add_argument(
        "--reference_audio_path", required=True, help="Path to reference audio file"
    )
    parser.add_argument(
        "--reference_text", required=True, help="Reference text for comparison"
    )
    parser.add_argument(
        "--output_path", required=True, help="Path where output TTS file will be saved"
    )
    args = parser.parse_args()

    generate_tts_file(
        args.key,
        args.input_text,
        args.reference_audio_path,
        args.reference_text,
        args.output_path,
    )
    print(f"out:{args.output_path} suc")
