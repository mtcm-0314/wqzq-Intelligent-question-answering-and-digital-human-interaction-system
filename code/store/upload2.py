from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np

# ------------------ 连接 Milvus ------------------ #
CLUSTER_ENDPOINT = "https://in03-4bb3b5dc9f774d4.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "358abb9a79b5795fc5a66d21be8a100fde909ce78c21e9918e7ee88a6abfc587ad47fdcd0e8184798b91c066a09c2207731d57cc"
connections.connect(alias="default", uri=CLUSTER_ENDPOINT, token=TOKEN)


# ------------------ 创建新集合（包含权重字段） ------------------ #
collection_name = "chatbot_collection"
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
weight_field = FieldSchema(name="weight", dtype=DataType.FLOAT)
schema = CollectionSchema(fields=[id_field, text_field, vector_field, weight_field])
collection = Collection(name=collection_name, schema=schema)

collection.load()

# ------------------ 插入示例数据（含权重） ------------------ #
def insert_data(user_input, user_embedding, weight: float = 1.0):
    entities = [
        [user_input],         # text
        [user_embedding],     # embedding
        [weight]              # weight
    ]
    collection.insert(entities)
    print("✅ 数据插入成功")

user_input = "覃锵是207的自行车高手，也是夜露高手"
user_embedding = np.random.rand(384).tolist()
weight_value = 1

insert_data(user_input, user_embedding, weight=weight_value)
