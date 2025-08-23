import requests


def test_ask_endpoint(question):
    url = "http://localhost:8000/ask"
    params = {"question": question}

    # 使用 stream=True 启用流式响应
    with requests.get(url, params=params, stream=True) as response:
        if response.status_code == 200:
            # 逐块读取响应内容
            for chunk in response.iter_content(chunk_size=128):
                if chunk:  # 确保 chunk 不为空
                    print(chunk.decode('utf-8'), end='', flush=True)  # 实时打印
        else:
            print(f"错误：状态码 {response.status_code} - {response.text}")


if __name__ == "__main__":
    # 示例问题
    # test_question = "分布式锁实现方式"
    # 先问的llm，生成答案，然后回答，确实查询准确率高了
    # test_question = """
    # 分布式锁是一种在分布式系统中控制对共享资源访问的技术，常见的实现方式包括：
    #
    # Redis：使用 Redis 的 SETNX（设置如果不存在）命令来实现原子锁操作，结合过期时间防止死锁。
    # Zookeeper：利用 Zookeeper 的临时顺序节点特性，通过抢锁的方式实现分布式锁。
    # 数据库乐观锁：在数据库中使用版本号字段实现乐观并发控制。
    # """
    test_question = "分布式锁常用的工具有哪些?"
    print(f"发送问题: {test_question}")
    test_ask_endpoint(test_question)