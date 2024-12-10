import base64
import requests
from minio import Minio
from io import BytesIO
from minio.error import S3Error
import random
from hashlib import md5

minio_url = "1.12xx0"
access_key = "K3SZxxhAdeKHFkC"
secret_key = "ErvvxcXsuByJDZpELcPFJL"
Authorization = "bearer 3iQITxxiDnXsy"


def send_inference_request(prompt):
    # 初始化 MinIO 客户端
    client = Minio(
        minio_url,
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )
    # 桶名称
    bucket_name = "top-ai-lch"
    # 创建桶
    try:
        # 检查桶是否存在
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"桶 '{bucket_name}' 创建成功！")
        else:
            print(f"桶 '{bucket_name}' 已连接！")
    except S3Error as e:
        print(f"创建桶时发生错误: {e}")

    def compute_mdhash_id(content, prefix: str = ""):
        return prefix + md5(content.encode()).hexdigest()

    url = "https://api.deepinfra.com/v1/inference/black-forest-labs/FLUX-1-dev"
    headers = {
        "Authorization": Authorization,
        "Content-Type": "application/json",
    }
    data = {"prompt": prompt, "width": 512, "height": 512}

    response = requests.post(url, headers=headers, json=data)
    random_number = random.randint(1, 10000)
    name = compute_mdhash_id(prompt.strip(), prefix="pic_")
    object_name = name + str(random_number) + ".jpg"

    if response.status_code == 200:
        response_data = response.json()
        image_data = response_data.get("images", [None])[0]
        base64_str = image_data.split(",")[1]
        # print(f"{base64_str}")
        image_bytes = base64.b64decode(base64_str)
        image_file = BytesIO(image_bytes)
        client.put_object(bucket_name, object_name, image_file, len(image_bytes))
        # 设置对象为公开读取
        url = client.presigned_get_object(bucket_name, object_name)
        public_url = f"http://{minio_url}/{bucket_name}/{object_name}"
        print(f"{url}\n{public_url}")
        return public_url
    else:
        response.raise_for_status()


# python test/usua2/get_flux_pics.py
prompt = "an old, twisted vine clinging to an ancient tree with a few crows resting on the branches, in a dim, fading light."
result = send_inference_request(prompt)
