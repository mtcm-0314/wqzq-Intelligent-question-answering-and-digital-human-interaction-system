from pymilvus import connections, Collection

# ------------------ 配置 ------------------ #
CLUSTER_ENDPOINT = "https://in03-4bb3b5dc9f774d4.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "358abb9a79b5795fc5a66d21be8a100fde909ce78c21e9918e7ee88a6abfc587ad47fdcd0e8184798b91c066a09c2207731d57cc"
COLLECTION_NAME = "chatbot_collection"

# ------------------ 连接 Milvus ------------------ #
try:
    connections.connect(alias="default", uri=CLUSTER_ENDPOINT, token=TOKEN)
    print("✅ 成功连接到 Milvus")
except Exception as e:
    print(f"❌ 无法连接 Milvus: {e}")
    exit()

# ------------------ 清空集合数据 ------------------ #
try:
    collection = Collection(name=COLLECTION_NAME)
    collection.delete(expr="id >= 0")  # 删除所有记录
    collection.load()
    print(f"✅ 集合 '{COLLECTION_NAME}' 的所有数据已清空")
except Exception as e:
    print(f"❌ 清空集合失败: {e}")
