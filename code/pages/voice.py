import streamlit as st
import pyttsx3  # 引入 pyttsx3
import requests  # 导入 requests
import re  # 导入正则表达式模块
import json
# 初始化 pyttsx3 引擎
engine = pyttsx3.init()

# 设置语速、音量等属性（可选）
engine.setProperty('rate', 150)  # 设置语速
engine.setProperty('volume', 1)  # 设置音量（0.0 到 1.0）

# 设置 Ollama API 地址
OLLAMA_API_URL = "http://localhost:11434/api/generate"

st.title("Ollama Chatbot 🤖")
st.write("一个可以记住 20 轮对话的 AI ！")

# 初始化 session_state 用于存储聊天记录，最多保留 20 轮
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# **先渲染已存的聊天记录，保证即时显示**
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# 用户输入框
user_input = st.chat_input("请输入你的问题...")

if user_input:
    # **立即显示用户输入**
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # **拼接历史对话，形成上下文**
    formatted_history = "\n".join(
        [f"{chat['role'].capitalize()}: {chat['content']}" for chat in st.session_state.chat_history]
    )

    # **请求 Ollama，开启流式响应**
    payload = {
        "model": "deepseek-r1",  # Ollama 模型
        "prompt": f"以下是你和用户的对话历史，请基于这些内容回答问题：\n{formatted_history}\nAssistant:",
        "stream": True  # 开启流式输出
    }

    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # **创建一个占位符，用于动态更新内容**
        full_response = ""

        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)

        if response.status_code == 200:
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        decoded_chunk = chunk.decode("utf-8")  # 解码数据
                        json_chunk = json.loads(decoded_chunk)  # 解析 JSON 响应

                        response_text = json_chunk.get('response', '').replace("\n", " ").strip()
                        if response_text:
                            full_response += response_text  # 逐步累积响应
                            message_placeholder.markdown(full_response + "▌")  # **实时更新文本**

                    except json.JSONDecodeError:
                        st.warning("收到无效的 JSON 数据，跳过该部分响应。")

            # **清理掉 <think> 和 </think> 之间的内容**
            cleaned_response = re.sub(r'<think>.*?</think>', '', full_response)
            message_placeholder.markdown(full_response)  # **去掉光标**

            # **存入聊天记录**
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            # **最多保留 20 轮**
            if len(st.session_state.chat_history) > 20:
                st.session_state.chat_history.pop(0)
            # **清理后的文本用于语音播报**
            engine.say(cleaned_response)
            engine.runAndWait()

        else:
            st.error(f"请求失败，状态码：{response.status_code}")
