import requests

headers = {"Authorization": "Bearer torch-bvisuwbksndclksjiocjwv742d"}
request_data = {
    "messages": [
        # {"role": "user", "content": "test-chat"} 此处仅图片理解,所以不需要chat
        {
            "role": "user",
            "content": [
                # {"type": "text", "text": "{'key1': 'value1', 'key2': 'value2'}"},
                {"type": "text", "text": "test-chat"},
                {
                    "type": "image_url",
                    "image_url": {"url": "no_git_oic/采购合同2.pdf_show_0.jpg"},
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.mfa.gov.cn/zwbd_673032/jghd_673046/202410/W020241008522924065946.jpg"
                    },
                },
            ],
        },
    ],
}
response = requests.post(
    "http://127.0.0.1:8121/v1/chat/completions", json=request_data, headers=headers
)

print(f"{response.json()['choices'][0]['message']['content']}")
# python test/litserve/client/openai_client.py
