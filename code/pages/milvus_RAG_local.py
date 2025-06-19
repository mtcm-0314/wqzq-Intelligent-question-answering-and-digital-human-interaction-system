import streamlit as st
import pyttsx3
import requests
import json
import threading
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from streamlit.components.v1 import html
from parts.voice_input_component import voice_input_component
from streamlit_js_eval import streamlit_js_eval
import numpy as np
# ------------------ 初始化 ------------------ #
live2d_js = """
<script src="https://fastly.jsdelivr.net/gh/stevenjoezhang/live2d-widget@latest/autoload.js"></script>
"""
html(live2d_js, height=400)

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

# ------------------ Streamlit UI ------------------ #
st.title("💬 DeepSeek + Ollama 本地问答机器人")
st.write("支持本地 Ollama 模型、Milvus 检索、流式响应、语音播放")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个乐于助人的AI助手，用中文简洁清晰地回答问题"}]

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ------------------ Milvus 初始化 ------------------ #
CLUSTER_ENDPOINT = "https://in03-4bb3b5dc9f774d4.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "358abb9a79b57207731d57cc"

try:
    connections.connect(alias="default", uri=CLUSTER_ENDPOINT, token=TOKEN)
except Exception as e:
    st.error(f"❌ 连接 Milvus 失败: {e}")
    st.stop()

collection_name = "guilin_qa_collection"
existing_collections = utility.list_collections()

if collection_name not in existing_collections:
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    # 修改 FieldSchema 定义（添加 FLOAT 类型的 weight 字段）
    weight_field = FieldSchema(name="weight", dtype=DataType.FLOAT)  # 权重字段
    schema = CollectionSchema(
        fields=[id_field, text_field, vector_field, weight_field]  # 加入 weight
    )
    # schema = CollectionSchema(fields=[id_field, text_field, vector_field])
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

# ------------------ 检索函数 ------------------ #
# def search_similar_docs(query_text, top_k=3):
#     query_vector = get_embedding(query_text)
#     search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
#     results = collection.search(
#         data=[query_vector],
#         anns_field="embedding",
#         param=search_params,
#         limit=top_k,
#         output_fields=["text"]
#     )
#     hits = results[0]
#     return [hit.entity.get("text") for hit in hits]
def search_similar_docs(query_text, top_k=3):
    query_vector = get_embedding(query_text)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k * 2,  # 多取一些结果用于后续加权
        output_fields=["text", "weight"]
    )

    hits = results[0]
    # 计算加权得分
    weighted_results = []
    for hit in hits:
        score = hit.score
        # st.write("score:", score)
        text = hit.entity.get("text")
        # st.write("text:", text)
        weight = hit.entity.get("weight")  # 默认权重1.0
        # st.write("weight:", weight)
        # weighted_score = score * weight
        weighted_score = score * (1 + np.log(weight + 1))
        weighted_results.append((text, weighted_score))

    # 按加权得分排序
    weighted_results.sort(key=lambda x: x[1], reverse=True)

    # 返回前top_k个结果
    return [result[0] for result in weighted_results[:top_k]]
# ------------------ 本地 OLLAMA 设置 ------------------ #
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1"

# ------------------ 用户输入 & 主逻辑 ------------------ #
# 在页面上加载语音识别组件
voice_input_component()

# 尝试获取识别结果
transcript = streamlit_js_eval(js_expressions="window._lastVoiceTranscript", key="voice_input")

if transcript and transcript.strip():
    st.session_state.voice_input = transcript.strip()
    st.experimental_rerun()

# 检查是否有语音输入内容
user_input = st.chat_input("请输入你的问题...") or st.session_state.pop("voice_input", "")

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
        related_docs = search_similar_docs(user_input, top_k=10)
        knowledge_context = "\n".join(related_docs)
        system_prompt = f"你是一个聪明的 AI 助手，请参考以下知识回答用户的问题：\n{knowledge_context}"
    except Exception as e:
        st.error(f"❌ Milvus 检索失败: {e}")
        st.stop()

    context_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages[-10:]

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

    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": True
        }
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=(10, 30))
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
        st.error(f"❌ Ollama 本地模型调用失败: {e}")

# ------------------ 停止朗读按钮 ------------------ #
if st.button("🛑 停止朗读"):
    stop_tts()
