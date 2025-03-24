import streamlit as st
import pyttsx3
import requests
import json
import threading

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
    stop_speaking.clear()  # 允许朗读

    engine = create_tts_engine()

    for word in text.split():
        if stop_speaking.is_set():
            break  # 立即停止朗读
        engine.say(word)
        engine.runAndWait()  # 播放当前单词，等待完成

    engine.stop()  # 确保朗读终止

# 停止朗读
def stop_tts():
    global speak_thread
    stop_speaking.set()  # 设置终止标志
    engine = create_tts_engine()
    engine.stop()  # 立即停止朗读

    if speak_thread and speak_thread.is_alive():
        speak_thread.join()  # 等待朗读线程终止
    speak_thread = None  # 清空线程

# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-aamgdovwgwalykadxfwdkbipuusdggapytopbblgihybnakn"
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

# 获取用户输入
user_input = st.chat_input("请输入你的问题...")

if user_input:
    stop_tts()  # 先停止当前语音播放

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
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

            # 启动新的 TTS 线程
            speak_thread = threading.Thread(target=speak, args=(full_response,))
            speak_thread.start()

    except requests.RequestException as e:
        st.error(f"请求失败: {str(e)}")

# 添加按钮控制语音播放
if st.button("停止朗读"):
    stop_tts()
