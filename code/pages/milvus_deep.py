import streamlit as st
import pyttsx3
import requests
import json
import threading
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 停止标志 & 线程
stop_speaking = threading.Event()
speak_thread = None


# 初始化 TTS
def create_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    return engine


# 朗读文本
def speak(text):
    global speak_thread
    stop_speaking.clear()
    engine = create_tts_engine()

    for word in text.split():
        if stop_speaking.is_set():
            break
        engine.say(word)
        engine.runAndWait()

    engine.stop()


# 停止朗读
def stop_tts():
    global speak_thread
    stop_speaking.set()
    engine = create_tts_engine()
    engine.stop()

    if speak_thread and speak_thread.is_alive():
        speak_thread.join()
    speak_thread = None


# DeepSeek API 配置
DEEPSEEK_API_URL = "htions"
API_KEY = "sk-l"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

st.title("DeepSeek Chatbot 🤖")
st.write("Powered by DeepSeek R1")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个乐于助人的AI助手，用中文简洁清晰地回答问题"}]

# 显示历史消息
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# 你的 Milvus 云端配置
CLUSTER_ENDPOINT = ""
TOKEN = ""

# 连接 Milvus
try:
    connections.connect(alias="default", uri=CLUSTER_ENDPOINT, token=TOKEN)
    print("✅ 成功连接到 Milvus！")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    exit(1)

# 检查是否已连接
if not connections.has_connection("default"):
    print("❌ 连接未建立，程序退出")
    exit(1)

# 定义集合名称
collection_name = "chatbot_collection"

# 获取已存在的集合
existing_collections = utility.list_collections()  # ✅ 直接使用 `utility`
if collection_name not in existing_collections:
    print(f"🛠️ {collection_name} 不存在，正在创建...")

    # 定义 Schema
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)

    schema = CollectionSchema(fields=[id_field, text_field, vector_field], description="Chatbot 数据")

    # 创建 Collection
    collection = Collection(name=collection_name, schema=schema)  # ✅ 直接创建 Collection
    print(f"✅ {collection_name} 创建成功！")
else:
    # 获取已存在的 Collection
    collection = Collection(name=collection_name)
    print(f"✅ 成功获取 Collection: {collection_name}")
print("成功连接到 Milvus 并初始化集合！")

# 获取用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    stop_tts()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 插入用户输入到 Milvus
    sample_embedding = np.random.rand(768).tolist()  # 生成随机向量
    insert_result = collection.insert([[user_input], [sample_embedding]])
    st.write("数据已存入 Milvus, 主键为:", insert_result.primary_keys)

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": st.session_state.messages[-20:],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=True, timeout=(10, 30))
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8').strip()
                    if decoded_chunk == "data: [DONE]":
                        break
                    if decoded_chunk.startswith("data: "):
                        try:
                            json_data = json.loads(decoded_chunk[6:])
                            delta = json_data["choices"][0].get("delta", {})
                            chunk_content = delta.get("content", "")
                            if chunk_content:
                                full_response += chunk_content
                                message_placeholder.markdown(full_response + "▌")
                        except json.JSONDecodeError:
                            st.error(f"JSON 解析失败: {decoded_chunk}")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            speak_thread = threading.Thread(target=speak, args=(full_response,))
            speak_thread.start()
    except requests.RequestException as e:
        st.error(f"请求失败: {str(e)}")

# 停止朗读按钮
if st.button("停止朗读"):
    stop_tts()
