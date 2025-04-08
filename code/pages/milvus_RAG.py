import streamlit as st
import pyttsx3
import requests
import json
import threading
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# ------------------ 初始化 ------------------ #
stop_speaking = threading.Event()
speak_thread = None

def create_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    return engine

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

def stop_tts():
    global speak_thread
    stop_speaking.set()
    engine = create_tts_engine()
    engine.stop()
    if speak_thread and speak_thread.is_alive():
        speak_thread.join()
    speak_thread = None

# ------------------ 嵌入模型 ------------------ #
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
def get_embedding(text):
    return embedding_model.encode([text])[0].tolist()

# ------------------ Streamlit 界面 ------------------ #
st.title("DeepSeek Chatbot 🤖")
st.write("Powered by DeepSeek + Milvus 向量知识库")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个乐于助人的AI助手，用中文简洁清晰地回答问题"}]

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ------------------ Milvus 初始化 ------------------ #
CLUSTER_ENDPOINT = ""
TOKEN = ""

try:
    connections.connect(alias="default", uri=CLUSTER_ENDPOINT, token=TOKEN)
except Exception as e:
    st.error(f"❌ 连接 Milvus 失败: {e}")
    st.stop()

collection_name = "chatbot_collection"
existing_collections = utility.list_collections()

if collection_name not in existing_collections:
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    schema = CollectionSchema(fields=[id_field, text_field, vector_field])
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# ------------------ 检索函数 ------------------ #
def search_similar_docs(query_text, top_k=3):
    query_vector = get_embedding(query_text)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )
    hits = results[0]
    return [hit.entity.get("text") for hit in hits]

# ------------------ 用户输入 & DeepSeek ------------------ #
DEEPSEEK_API_URL = ""
API_KEY = ""
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

user_input = st.chat_input("请输入你的问题...")

if user_input:
    stop_tts()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 👉 向量插入
    user_embedding = get_embedding(user_input)
    insert_result = collection.insert([[user_input], [user_embedding]])
    st.write("🔗 已存入 Milvus, 主键为:", insert_result.primary_keys)

    # 🔍 检索上下文
    related_docs = search_similar_docs(user_input, top_k=3)
    knowledge_context = "\n".join(related_docs)
    system_prompt = f"你是一个聪明的 AI 助手，请参考以下知识回答用户的问题：\n{knowledge_context}"
    context_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-10:]

    # 🚀 调用 DeepSeek 接口
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": context_messages,
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

# ------------------ 停止朗读按钮 ------------------ #
if st.button("停止朗读"):
    stop_tts()
