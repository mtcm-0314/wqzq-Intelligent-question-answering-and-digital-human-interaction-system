import streamlit as st
import pyttsx3
import requests
import json
import threading
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
embedding_model = SentenceTransformer(r'C:\Users\mtcm\Desktop\东盟杯\code\local_embedding_model')

def get_embedding(text):
    return embedding_model.encode([text])[0].tolist()

# ------------------ Milvus 初始化 ------------------ #
CLUSTER_ENDPOINT = "https://in03-4bb3b5dc9f774d4.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "358abb9a79b5795fc5a66d21be8a100fde909ce78c21e9918e7ee88a6abfc587ad47fdcd0e8184798b91c066a09c2207731d57cc"

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
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    schema = CollectionSchema(fields=[id_field, text_field, vector_field])
    collection = Collection(name=collection_name, schema=schema)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
else:
    collection = Collection(name=collection_name)

collection.load()

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

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="本地问答机器人", page_icon="🤖")
st.title("💬 DeepSeek + Ollama 本地问答机器人")
st.write("支持本地 Ollama、DeepSeek API、Milvus 检索、流式响应、语音播放")

# 模型选择
st.sidebar.title("模型设置")
model_choice = st.sidebar.radio("选择使用的模型：", options=["本地 Ollama", "DeepSeek API"], index=0)

# 消息初始化
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个乐于助人的AI助手，用中文简洁清晰地回答问题"}]

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ------------------ 用户输入 & 主逻辑 ------------------ #
user_input = st.chat_input("请输入你的问题...")

if user_input:
    stop_tts()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        user_embedding = get_embedding(user_input)
    except Exception as e:
        st.error(f"❌ 获取嵌入失败: {e}")
        st.stop()

    try:
        related_docs = search_similar_docs(user_input, top_k=3)
        knowledge_context = "\n".join(related_docs)
        # 构造用户提问的上下文
        enriched_user_question = f"以下是一些可能相关的资料：\n{knowledge_context}\n\n现在请根据这些资料回答我的问题：{user_input}"
    except Exception as e:
        st.error(f"❌ Milvus 检索失败: {e}")
        st.stop()

    context_messages = [{"role": "system", "content": "你是一个乐于助人的 AI 助手。"}]
    context_messages.append({"role": "user", "content": enriched_user_question})

    # 模型调用部分
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if model_choice == "DeepSeek API":
                API_URL = "https://api.siliconflow.cn/v1/chat/completions"
                API_KEY = "sk-aamgdovwgwalykadxfwdkbipuusdggapytopbblgihybnakn"
                MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                }

                messages_payload = [{"role": msg["role"], "content": msg["content"]} for msg in context_messages]
                payload = {
                    "model": MODEL_NAME,
                    "messages": messages_payload,
                    "stream": True,
                    "temperature": 0.7,
                    "top_p": 0.9
                }

                response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=(10, 60))
                for line in response.iter_lines():
                    if line:
                        if line.startswith(b"data: "):
                            data_str = line[len(b"data: "):].decode("utf-8")
                            if data_str.strip() == "[DONE]":
                                break
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                            if isinstance(delta, str):
                                full_response += delta
                                message_placeholder.markdown(full_response + "▌")
            else:
                API_URL = "http://localhost:11434/api/generate"
                MODEL_NAME = "deepseek-r1"

                full_prompt = ""
                for msg in context_messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        full_prompt += f"用户: {content}\n"
                    elif role == "assistant":
                        full_prompt += f"助手: {content}\n"
                    elif role == "system":
                        full_prompt += f"{content}\n"

                payload = {
                    "model": MODEL_NAME,
                    "prompt": full_prompt,
                    "stream": True
                }

                response = requests.post(API_URL, json=payload, stream=True, timeout=(10, 30))
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode("utf-8"))
                        content = data.get("response", "")
                        full_response += content
                        message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            speak_thread = threading.Thread(target=speak, args=(full_response,))
            speak_thread.start()

    except Exception as e:
        st.error(f"❌ 模型调用失败: {e}")

# ------------------ 停止朗读按钮 ------------------ #
if st.button("🛑 停止朗读"):
    stop_tts()
