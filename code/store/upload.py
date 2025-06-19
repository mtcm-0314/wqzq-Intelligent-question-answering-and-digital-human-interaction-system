from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(r'C:\Users\mtcm\Desktop\东盟杯\code\local_embedding_model')

def get_embedding(text):
    return embedding_model.encode([text])[0].tolist()
# 连接 Milvus
CLUSTER_ENDPOINT = "https://in03-4bb3b5dc9f774d4.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "358abb9a79b5795fc5a66d21be8a100fde909ce78c21e9918e7ee88a6abfc587ad47fdcd0e8184798b91c066a09c2207731d57cc"
connections.connect(alias="default", uri=CLUSTER_ENDPOINT, token=TOKEN)

# 定义集合的名称
collection_name = "chatbot_collection1"

# 检查集合是否存在
if collection_name in utility.list_collections():
    collection = Collection(name=collection_name)
else:
    # 如果集合不存在，则创建集合
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # 确保 dim 与你嵌入向量的维度一致
    schema = CollectionSchema(fields=[id_field, text_field, vector_field])

    collection = Collection(name=collection_name, schema=schema)


# 插入数据
def insert_data(user_input, user_embedding):
    # 用户输入和嵌入向量
    # user_input: 文本数据
    # user_embedding: 对应的 384 维度的浮动向量
    entities = [
        [user_input],  # 文本字段
        [user_embedding]  # 嵌入向量字段
    ]
    # 批量插入数据
    collection.insert(entities)
    print("数据已成功插入")


# 示例：插入用户数据
user_input = ""

# 生成一个模拟的 384 维度嵌入向量
# user_embedding = np.random.rand(384).tolist()  # 这里使用了随机数生成示例向量，实际情况应使用嵌入模型生成
user_embedding=get_embedding(user_input)

# 插入数据
insert_data(user_input, user_embedding)
