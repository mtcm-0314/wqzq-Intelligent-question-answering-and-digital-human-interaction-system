import streamlit as st
import pyttsx3
import requests
import json
import threading
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from streamlit.components.v1 import html
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import io
import numpy as np
import re
import time
import streamlit.components.v1 as components
import edge_tts
import sounddevice as sd
import soundfile as sf
import asyncio
import io
import threading
import fitz  # PyMuPDF
import atexit
import docx
# ------------------ 页面设置 ------------------ #
st.set_page_config(
    page_title="DeepSeek + Ollama 本地问答机器人",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ CSS样式 ------------------ #
st.markdown("""
<style>
    /* 主容器样式 */
    .stApp {
        background-color: #f5f7fa;
    }

    /* 聊天消息样式 */
    .stChatMessage {
        border-radius: 15px;
        padding: 12px 18px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* 用户消息样式 */
    [data-testid="stChatMessage-user"] {
        background-color: #e3f2fd;
        margin-left: 20%;
    }

    /* 助手消息样式 */
    [data-testid="stChatMessage-assistant"] {
        background-color: #ffffff;
        margin-right: 20%;
    }

    /* 按钮样式 */
    .stButton>button {
        border-radius: 20px;
        padding: 8px 16px;
        background-color: #4a6fa5;
        color: white;
        border: none;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #3a5a80;
        transform: scale(1.05);
    }

    /* 语音按钮特殊样式 */
    #voice_button_side {
        background-color: #ff6b6b;
    }

    #voice_button_side:hover {
        background-color: #ff5252;
    }

    /* 输入框样式 */
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px 15px;
    }

    /* 状态提示样式 */
    .status-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 10px 0;
        border-left: 4px solid #4a6fa5;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ 初始化 ------------------ #
# ------------------ 添加 Live2D 看板娘 ------------------ #
live2d_js = """
<script src="https://fastly.jsdelivr.net/gh/mtcm-0314/live2d-widget@latest/dist/autoload.js"></script>
"""
html(live2d_js, height=400)



stop_speaking = threading.Event()
speak_thread = None


# ------------------ 语音识别模型 ------------------ #
@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", device="cpu", compute_type="int8")


whisper_model = load_whisper_model()


# ------------------ 语音功能 ------------------ #
stop_speaking = threading.Event()
speak_thread = None

# Edge-TTS 支持的中文语音选项
voice_options = {
    "女声（晓晓）": "zh-CN-XiaoxiaoNeural",
    "男声（云希）": "zh-CN-YunxiNeural",
    "童声（云夏）": "zh-CN-YunxiaNeural",
    "情感 (小毅）": "zh-CN-XiaoyiNeural",
    "正式（云杨）":"zh-CN-YunyangNeural"
}

voice_name = st.selectbox("选择语音角色", list(voice_options.keys()), index=0)
voice = voice_options[voice_name]

def stop_tts():
    global speak_thread
    stop_speaking.set()
    sd.stop()
    if speak_thread and speak_thread.is_alive():
        speak_thread.join()
    speak_thread = None

def speak(text, voice=voice):
    global speak_thread
    stop_speaking.clear()

    async def run_tts():
        try:
            communicate = edge_tts.Communicate(text, voice)
            audio_stream = communicate.stream()
            audio_bytes = b""
            async for chunk in audio_stream:
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            audio_array, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if not stop_speaking.is_set():
                sd.play(audio_array, samplerate)
                sd.wait()
        except Exception as e:
            st.error(f"❌ 语音合成错误: {e}")

    asyncio.run(run_tts())


# ------------------ 嵌入模型 ------------------ #
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(r'C:\Users\mtcm\Desktop\东盟杯\code\local_embedding_model')


embedding_model = load_embedding_model()


def get_embedding(text):
    return embedding_model.encode([text])[0].tolist()


# ------------------ 应用标题 ------------------ #
st.title("🤖 DeepSeek + Ollama 本地问答机器人")
st.markdown("""
<script>
window.addEventListener("DOMContentLoaded", function () {
    const waitInput = setInterval(() => {
        const inputBox = parent.document.querySelector('textarea[data-testid="stChatInput"]');
        const waifuTips = parent.document.getElementById('waifu-tips');

        if (inputBox && waifuTips) {
            clearInterval(waitInput);

            inputBox.addEventListener('input', () => {
                const text = inputBox.value.trim();
                if (text !== "") {
                    const thinkingPhrases = ["正在思考中...", "让我想想...", "稍等片刻～", "这个问题有点难呢...", "让我翻翻笔记..."];
                    const msg = thinkingPhrases[Math.floor(Math.random() * thinkingPhrases.length)];

                    waifuTips.innerHTML = msg;
                    waifuTips.classList.add("waifu-tips-active");
                }
            });

            inputBox.addEventListener('blur', () => {
                waifuTips.innerHTML = '';
                waifuTips.classList.remove("waifu-tips-active");
            });
        }
    }, 500);
});
</script>
""", unsafe_allow_html=True)



# ------------------ 聊天历史 ------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个乐于助人的AI助手，用中文简洁清晰地回答问题"}]

# 显示聊天历史
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ------------------ Milvus 初始化 ------------------ #
CLUSTER_ENDPOINT = "https://in03-4bb3b5dc9f774d4.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "358abb9a7807731d57cc"

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
    weight_field = FieldSchema(name="weight", dtype=DataType.FLOAT)
    schema = CollectionSchema(
        fields=[id_field, text_field, vector_field, weight_field]
    )
    collection = Collection(name=collection_name, schema=schema)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
else:
    collection = Collection(name=collection_name)


# ----------- 文本提取函数 ----------- #
def extract_text_from_file(file, file_type):
    if file_type == "pdf":
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    elif file_type == "docx":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_type == "txt":
        return file.read().decode("utf-8")

    else:
        return ""

# ----------- 分块函数 ----------- #
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ----------- 自动删除上传向量 ----------- #
def delete_uploaded_vectors():
    if "uploaded_vector_ids" in st.session_state:
        try:
            ids = st.session_state.uploaded_vector_ids
            id_expr = f"id in [{','.join(map(str, ids))}]"
            collection.delete(expr=id_expr)
            st.info(f"🧹 自动删除了 {len(ids)} 条向量。")
        except Exception as e:
            st.warning(f"❌ 删除向量失败: {e}")

atexit.register(delete_uploaded_vectors)

# ----------- 文件上传 UI ----------- #
uploaded_file = st.file_uploader("📄 上传文件 (支持 PDF / Word / TXT)", type=["pdf", "docx", "txt"])
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    st.success(f"📤 文件 {uploaded_file.name} 上传成功，类型：{file_type.upper()}")

    with st.spinner("📄 正在提取文本..."):
        text = extract_text_from_file(uploaded_file, file_type)

    if text.strip() == "":
        st.warning("⚠️ 文件中没有提取到有效文本")
    else:
        chunks = chunk_text(text)
        st.info(f"📌 提取到 {len(chunks)} 个段落，正在生成向量并写入数据库...")

        try:
            vectors = [get_embedding(chunk) for chunk in chunks]
            insert_result = collection.insert([
                chunks,
                vectors,
                [1.0] * len(chunks)
            ])
            inserted_ids = insert_result.primary_keys
            st.session_state.uploaded_vector_ids = inserted_ids
            st.success(f"✅ 插入成功：{len(inserted_ids)} 条向量，退出后将自动删除。")
        except Exception as e:
            st.error(f"❌ 向量插入失败: {e}")
# ------------------ 检索函数 ------------------ #
def search_similar_docs(query_text, top_k=3):
    query_vector = get_embedding(query_text)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k * 2,
        output_fields=["text", "weight"]
    )

    hits = results[0]
    weighted_results = []
    for hit in hits:
        score = hit.score
        text = hit.entity.get("text")
        weight = hit.entity.get("weight")
        weighted_score = score * (1 + np.log(weight + 1))
        weighted_results.append((text, weighted_score))

    weighted_results.sort(key=lambda x: x[1], reverse=True)
    return [result[0] for result in weighted_results[:top_k]]


# ------------------ 本地 OLLAMA 设置 ------------------ #
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1"

# ------------------ 输入区域 ------------------ #
# 创建两列布局：输入框和语音按钮
input_col, voice_col = st.columns([5, 1])

with input_col:
    # 初始化聊天输入值
    if "chat_input_value" not in st.session_state:
        st.session_state.chat_input_value = ""
    components.html(
        """
        <div id="waifu-streamlit-input-box"></div>
        """,
        height=0,
    )
    # 创建聊天输入框
    user_input = st.chat_input(
        "请输入你的问题...",
        key="chat_input"
    )

with voice_col:
    # 语音按钮
    if st.button("🎤", key="voice_button_side", help="点击开始语音输入"):
        st.session_state.recording = True
        st.session_state.voice_input_result = ""
        st.rerun()

# 语音录制和处理
if st.session_state.get("recording", False):
    with st.container():
        st.markdown('<div class="status-box">🎤 正在录音... (说话后自动停止)</div>', unsafe_allow_html=True)
        audio_bytes = audio_recorder(text="", pause_threshold=2.0, key="audio_recorder")

        if audio_bytes:
            st.session_state.recording = False
            with st.spinner("🔍 正在识别语音..."):
                recognized_text = transcribe_audio(audio_bytes)
                if recognized_text:
                    st.session_state.chat_input_value = recognized_text
                    st.rerun()

# 如果会话状态中有预设值，使用它作为用户输入
if "chat_input_value" in st.session_state and st.session_state.chat_input_value:
    user_input = st.session_state.chat_input_value
    st.session_state.chat_input_value = ""

# ------------------ 处理用户输入 ------------------ #
if user_input:
    stop_tts()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        related_docs = search_similar_docs(user_input, top_k=10)
        st.write("related_docs",related_docs)
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
            clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL)
            speak_thread = threading.Thread(target=speak, args=(clean_response,))
            speak_thread.start()

    except Exception as e:
        st.error(f"❌ Ollama 本地模型调用失败: {e}")

# ------------------ 底部控制按钮 ------------------ #
button_col1, button_col2 = st.columns([1, 1])

with button_col1:
    if st.button("🛑 停止朗读", help="停止当前语音朗读"):
        stop_tts()

with button_col2:
    if st.button("🧹 清除对话", help="清除所有聊天历史"):
        st.session_state.messages = [{"role": "system", "content": "你是一个乐于助人的AI助手，用中文简洁清晰地回答问题"}]
        st.rerun()
