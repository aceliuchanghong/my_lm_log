import asyncio


# python test/usua2/sync_test.py
# 定义异步函数
async def fetch_data():
    print("开始获取数据...")
    await asyncio.sleep(5)  # 模拟网络请求，暂停5秒
    print("数据获取完成")
    return "数据内容"


async def main():
    print("程序开始")
    result = await fetch_data()  # 等待 fetch_data 完成
    print(f"获取到的数据: {result}")


# 运行异步任务
asyncio.run(main())
