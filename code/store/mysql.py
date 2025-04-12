from pymilvus import MilvusClient, DataType

CLUSTER_ENDPOINT = "https://in03-4bb3b5dc9f774d4.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "35c2207731d57cc"

# 1. Set up a Milvus client
client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN
)
# 2. Create a collection in quick setup mode
client.create_collection(
    collection_name="quick_setup",
    dimension=5
)
# 3. Create a collection in customized setup mode

# 3.1. Create schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

# 3.2. Add fields to schema
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)

# 3.3. Prepare index parameters
index_params = client.prepare_index_params()

# 3.4. Add indexes
index_params.add_index(
    field_name="my_id"
)

index_params.add_index(
    field_name="my_vector",
    index_type="AUTOINDEX",
    metric_type="IP"
)

# 3.5. Create a collection
client.create_collection(
    collection_name="customized_setup",
    schema=schema,
    index_params=index_params
)
# 4. Insert data into the collection
# 4.1. Prepare data
data=[
    {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "pink_8682"},
    {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "color": "red_7025"},
    {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], "color": "orange_6781"},
    {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345], "color": "pink_9298"},
    {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106], "color": "red_4794"},
    {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955], "color": "yellow_4222"},
    {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987], "color": "red_9392"},
    {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052], "color": "grey_8510"},
    {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336], "color": "white_9381"},
    {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608], "color": "purple_4976"}
]

# 4.2. Insert data
res = client.insert(
    collection_name="quick_setup",
    data=data
)

print(res)

# Output
#
# {
#     "insert_count": 10,
#     "ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# }
import time
import random
# 5. Insert more data into the collection
# 5.1. Prepare data

colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "brown", "grey"]
data = [ {
    "id": i,
    "vector": [ random.uniform(-1, 1) for _ in range(5) ],
    "color": f"{random.choice(colors)}_{str(random.randint(1000, 9999))}"
} for i in range(1000) ]

# 5.2. Insert data
res = client.insert(
    collection_name="quick_setup",
    data=data[10:]
)

print(res)

# Output
#
# {
#     "insert_count": 990
# }

# Wait for a while
time.sleep(5)
# 6.1. Prepare query vectors
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648]
]

# 6.2. Start search
res = client.search(
    collection_name="quick_setup",     # target collection
    data=query_vectors,                # query vectors
    limit=3,                           # number of returned entities
)

print(res)

# Output
#
# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {}
#         },
#         {
#             "id": 296,
#             "distance": 0.0800950899720192,
#             "entity": {}
#         },
#         {
#             "id": 43,
#             "distance": 0.07794742286205292,
#             "entity": {}
#         }
#     ]
# ]
# 7. Search with multiple vectors
# 7.1. Prepare query vectors
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648],
    [0.0039737443, 0.003020432, -0.0006188639, 0.03913546, -0.00089768134]
]

# 7.2. Start search
res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    limit=3,
)

print(res)

# Output
#
# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {}
#         },
#         {
#             "id": 296,
#             "distance": 0.0800950899720192,
#             "entity": {}
#         },
#         {
#             "id": 43,
#             "distance": 0.07794742286205292,
#             "entity": {}
#         }
#     ],
#     [
#         {
#             "id": 730,
#             "distance": 0.04431751370429993,
#             "entity": {}
#         },
#         {
#             "id": 333,
#             "distance": 0.04231833666563034,
#             "entity": {}
#         },
#         {
#             "id": 232,
#             "distance": 0.04221535101532936,
#             "entity": {}
#         }
#     ]
# ]
# 8. Search with a filter expression using schema-defined fields
# 1 Prepare query vectors
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648]
]

# 2. Start search
res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    filter="500 < id < 800",
    limit=3
)

print(res)

# Output
#
# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {}
#         },
#         {
#             "id": 760,
#             "distance": 0.07432225346565247,
#             "entity": {}
#         },
#         {
#             "id": 539,
#             "distance": 0.07279646396636963,
#             "entity": {}
#         }
#     ]
# ]
# 9. Search with a filter expression using custom fields
# 9.1.Prepare query vectors
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648]
]

# 9.2.Start search
res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    filter='$meta["color"] like "red%"',
    limit=3,
    output_fields=["color"]
)

print(res)

# Output
#
# [
#     [
#         {
#             "id": 263,
#             "distance": 0.0744686871767044,
#             "entity": {
#                 "color": "red_9369"
#             }
#         },
#         {
#             "id": 381,
#             "distance": 0.06509696692228317,
#             "entity": {
#                 "color": "red_9315"
#             }
#         },
#         {
#             "id": 360,
#             "distance": 0.057343415915966034,
#             "entity": {
#                 "color": "red_6066"
#             }
#         }
#     ]
# ]
# 10. Query with a filter expression using a schema-defined field
res = client.query(
    collection_name="quick_setup",
    filter="10 < id < 15",
    output_fields=["color"]
)

print(res)

# Output
#
# [
#     {
#         "color": "yellow_4104",
#         "id": 11
#     },
#     {
#         "color": "blue_7278",
#         "id": 12
#     },
#     {
#         "color": "orange_7136",
#         "id": 13
#     },
#     {
#         "color": "pink_7776",
#         "id": 14
#     }
# ]
# 11. Query with a filter expression using a custom field
res = client.query(
    collection_name="quick_setup",
    filter='$meta["color"] like "brown_8%"',
    output_fields=["color"],
    limit=5
)

print(res)

# Output
#
# [
#     {
#         "color": "brown_8454",
#         "id": 17
#     },
#     {
#         "color": "brown_8390",
#         "id": 35
#     },
#     {
#         "color": "brown_8442",
#         "id": 309
#     },
#     {
#         "color": "brown_8429",
#         "id": 468
#     },
#     {
#         "color": "brown_8020",
#         "id": 472
#     }
# ]
# 13. Delete entities by IDs
res = client.delete(
    collection_name="quick_setup",
    ids=[0,1,2,3,4]
)

print(res)

# Output
#
# {
#     "delete_count": 5
# }
# 14. Delete entities by a filter expression
res = client.delete(
    collection_name="quick_setup",
    filter="id in [5,6,7,8,9]"
)

print(res)

# Output
#
# {
#     "delete_count": 5
# }
# 15. Drop collection
client.drop_collection(
    collection_name="quick_setup"
)

client.drop_collection(
    collection_name="customized_setup"
)
