import requests

headers = {"Authorization": "Bearer torch-bvisuwbksndclksjiocjwv742d"}
request_data = {
    # "model": "my-gpt2",
    "messages": [
        {"role": "user", "content": "helolo"},
    ],
}
response = requests.post(
    "http://127.0.0.1:8121/v1/chat/completions", json=request_data, headers=headers
)
# python test/litserve/client/openai_client.py
# print(f"{response.text}")
print(f"{response.json()['choices'][0]['message']['content']}")
