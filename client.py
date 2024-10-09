import requests

response = requests.post(
    "http://127.0.0.1:8109/predict",
    json={
        "images_path": ['./z_using_files/pics/00.png', './z_using_files/pics/11.jpg',
                        './z_using_files/pics/00006737.jpg','https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg'],
        "table": "normal"
    }
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
