import requests
import base64
import argparse


def generate_image(prompt: str, output_path: str = "output_image.png"):
    url = "https://api.deepinfra.com/v1/inference/stabilityai/sd3.5"
    headers = {
        "Authorization": "bearer 3iQIT7UW994mUKVLLmJDLEI4aJiDnXsy",
        "Content-Type": "application/json",
    }
    data = {"prompt": prompt}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()

        # 获取Base64编码的图片数据，并去掉前缀
        image_base64 = result["images"][0].split(",")[1]

        # 解码并保存图片
        image_data = base64.b64decode(image_base64)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"图片已保存为 {output_path}")
    else:
        print("请求失败，状态码：", response.status_code)


if __name__ == "__main__":
    """
    python test/litserve/api/gen_pic_server.py --prompt "A serene night scene featuring the ancient Chinese poet Li Bai gazing up at a luminous full moon. The surroundings are calm, with soft moonlight casting a gentle glow over a quiet landscape. Li Bai stands in traditional robes, exuding a sense of contemplation and wonder under the vast, star-filled sky."
    """
    parser = argparse.ArgumentParser(description="图像生成")
    parser.add_argument(
        "--prompt",
        default="A young shepherd boy standing on a hill, pointing towards a distant village surrounded by blossoming apricot trees, bathed in the soft glow of early morning sunlight. The village is nestled in a peaceful valley, with rolling hills in the background, and delicate white apricot flowers dotting the landscape. The scene conveys a serene and idyllic rural atmosphere, with a touch of nostalgia.",
        help="图片prompt",
    )
    args = parser.parse_args()
    generate_image(args.prompt)
