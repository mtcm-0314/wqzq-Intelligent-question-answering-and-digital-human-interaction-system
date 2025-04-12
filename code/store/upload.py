from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np

# 连接 Milvus
CLUSTER_ENDPOINT = "https://in03-4bb3b5dc9f774d4.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "358ab731d57cc"
connections.connect(alias="default", uri=CLUSTER_ENDPOINT, token=TOKEN)

# 定义集合的 schema
collection_name = "chatbot_collection"
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # 确保 dim 与你嵌入向量的维度一致
schema = CollectionSchema(fields=[id_field, text_field, vector_field])

# 加载已存在的集合（不重新创建）
collection = Collection(name=collection_name)

# 插入数据
def insert_data(user_input, user_embedding):
    # 用户输入和嵌入向量
    # user_input: 文本数据
    # user_embedding: 对应的 384 维度的浮动向量
    entities = [
        [user_input],        # 文本字段
        [user_embedding]     # 嵌入向量字段
    ]
    # 批量插入数据
    collection.insert(entities)
    print("数据已成功插入")

# 示例：插入用户数据
user_input = "插入数据，比如wqzq（完全正确）的队伍名是由四名队员的名字首字母组成的。"

# 生成一个模拟的 384 维度嵌入向量
user_embedding = np.random.rand(384).tolist()  # 这里使用了随机数生成示例向量，实际情况应使用嵌入模型生成

# 插入数据
insert_data(user_input, user_embedding)
