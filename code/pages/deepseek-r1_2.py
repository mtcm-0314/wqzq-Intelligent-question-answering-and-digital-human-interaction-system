import streamlit as st
import pyttsx3
import requests
import re
import json
import threading
from requests.exceptions import RequestException

# 移除了全局引擎初始化

# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "就那个"

st.title("DeepSeek Chatbot 🤖")
st.write("Powered by DeepSeek R1")

# 初始化对话历史（保留最多20轮）
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个乐于助人的AI助手，用中文简洁清晰地回答问题"}
    ]

# 渲染历史消息
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 用户输入处理
user_input = st.chat_input("请输入你的问题...")


def tts_worker(text):
    """独立的语音合成线程"""
    try:
        # 在线程内独立初始化引擎
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.session_state.tts_error = f"语音合成失败: {str(e)}"


if user_input:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 构造API请求
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "messages": st.session_state.messages[-20:],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    with st.chat_message("assistant"):
        try:
            # API请求处理
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)

            if response.status_code != 200:
                error_msg = f"API请求失败（状态码 {response.status_code}）"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('message', '未知错误')}"
                except:
                    error_msg += f"\n响应内容：{response.text[:200]}"
                raise RequestException(error_msg)

            response_data = response.json()
            full_response = response_data["choices"][0]["message"]["content"]
            cleaned_response = re.sub(r'<think>.*?</think>', '', full_response)

            # 显示并保存回复
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # 启动独立语音线程
            threading.Thread(target=tts_worker, args=(cleaned_response,), daemon=True).start()

        except Exception as e:
            error_mapping = {
                RequestException: lambda e: str(e),
                requests.SSLError: lambda e: f"SSL错误：{e}\n请检查系统证书",
                requests.Timeout: "请求超时，请检查网络连接",
                requests.ConnectionError: "连接服务器失败",
                KeyError: "API响应格式异常",
                json.JSONDecodeError: "无效的API响应格式"
            }
            error_msg = error_mapping.get(type(e), lambda x: f"未知错误：{x}")(e)
            st.error(error_msg)

    # 控制历史记录长度
    st.session_state.messages = st.session_state.messages[-20:]

# 显示语音错误（主线程处理）
if 'tts_error' in st.session_state:
    st.error(st.session_state.tts_error)
    del st.session_state.tts_error  # 显示后清除
