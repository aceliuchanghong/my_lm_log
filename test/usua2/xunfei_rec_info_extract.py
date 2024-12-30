def extract_info(message: str) -> dict:
    # 定义分隔符
    delimiter = "<|>"

    # 使用分隔符分割字符串
    parts = message.split(delimiter)

    if len(parts) != 2:
        raise ValueError(
            f"Err-info:整体输入的格式不正确,分隔符 {delimiter} 无法获取正确的2部分数据"
        )

    # 提取用户信息和聊天信息
    user_info, prompt = parts

    # 再次使用 '+' 分隔用户信息
    user_info_delimiter = "+"
    user_parts = user_info.split(user_info_delimiter)

    if len(user_parts) != 2:
        raise ValueError(
            f"Err-user:用户信息格式不正确,分隔符 {user_info_delimiter} 无法获取正确的2部分数据"
        )

    # 提取 username 和 userid
    username, userid = user_parts

    return {"username": username, "userid": userid, "prompt": prompt}


def rule_judge(info: dict) -> int:
    user_info = {"0000001": 1, "0000002": 2, "0000003": 3}
    level = user_info.get(info["userid"], 4)
    return level


def get_prompt(info: dict) -> str:
    return info["prompt"]


# str--json解析  ==> z_utils/get_json.py


def get_is_nsfw(info: dict) -> str:
    return info["is_nsfw"]


def get_is_real_time(info: dict) -> str:
    return info["is_real_time"]


# 示例用法
messages = ["雷xx+1550728<|>今天上海天气如何?", "刘XX+1551728<|>你是谁?"]

for message in messages:
    info = extract_info(message)
    level = rule_judge(info)
    print(
        f"username: {info['username']}, userid: {info['userid']}, prompt: {info['prompt']} level: {level}"
    )
"""
python test/usua2/xunfei_rec_info_extract.py

收到的信息,格式如下:
```
[用户信息][分隔符][用户发送的聊天信息]
用户信息=username+userid
分隔符=<|>
用户发送的聊天信息=prompt
example1:
雷xx+1552728<|>今天上海天气如何?
example2:
刘XX+1551728<|>你是谁?
```
游客+0000000
"""
