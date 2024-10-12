from fish_audio_sdk import Session, TTSRequest, ReferenceAudio


def generate_tts_file(session_id, input_text, reference_audio_path, output_path):
    session = Session(session_id)

    with open(reference_audio_path, "rb") as audio_file:
        with open(output_path, "wb") as f:
            for chunk in session.tts(
                TTSRequest(
                    text=input_text,
                    references=[
                        ReferenceAudio(
                            audio=audio_file.read(),
                            text="清晨温柔的光缓缓卸下,亲闭双眼,感觉那光像柔软的手在鼻尖轻敲,满是母亲的宠溺,张开双手,阳光便长绕了我,一点点一滴滴地融入我的身体,使我的血液都变得温热起来。",
                        )
                    ],
                )
            ):
                f.write(chunk)


# 调用函数示例
generate_tts_file(
    "233caaxxxx45eac26e905",
    "外观检查：应无机械损伤、起层或内电极裸露现象,安装用表面的每个棱边的熔蚀应不超过 25%(见图1)。",
    "z_using_files/mp3/ylh.wav",
    "no_git_oic/ylh_gen.mp3",
)
