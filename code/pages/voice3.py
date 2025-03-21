import streamlit as st
import pyttsx3
import requests
import re
import json
from collections import deque
import threading
import logging
import os

# ================== 全局配置 ==================
_TTS_ENGINE = None
_TTS_LOCK = threading.Lock()
_TTS_ERROR_LOG = "tts_errors.log"

# ============== 自定义样式 ==============
st.markdown("""
<style>
.think-text {
    color: #666666;
    font-style: italic;
    border-left: 3px solid #e0e0e0;
    padding-left: 10px;
    margin: 8px 0;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)


# ============== 初始化函数 ==============
def init_tts_engine():
    global _TTS_ENGINE
    try:
        if _TTS_ENGINE is None:
            _TTS_ENGINE = pyttsx3.init()
            _TTS_ENGINE.setProperty('rate', 150)
            _TTS_ENGINE.setProperty('volume', 1)
            logging.info("语音引擎初始化成功")

            # 测试语音
            _TTS_ENGINE.say("系统就绪")
            _TTS_ENGINE.runAndWait()
            _TTS_ENGINE.endLoop()
    except Exception as e:
        logging.error(f"语音初始化失败: {str(e)}")


if os.environ.get('STREAMLIT_RUNNING'):
    init_tts_engine()


# ============== 功能函数 ==============
def format_think_content(text):
    """格式化思考内容为带样式的HTML"""
    return re.sub(
        r'<think>(.*?)</think>',
        r'<div class="think-text">\1</div>',
        text,
        flags=re.DOTALL
    )


def tts_speak(text):
    """语音播报"""
    if not text:
        return

    try:
        with _TTS_LOCK:
            if _TTS_ENGINE is None:
                init_tts_engine()

            if _TTS_ENGINE:
                cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                _TTS_ENGINE.say(cleaned.strip())
                _TTS_ENGINE.runAndWait()
    except Exception as e:
        error_msg = f"语音错误: {str(e)}"
        logging.error(error_msg)
        with open(_TTS_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{error_msg}\n")


# ============== 页面组件 ==============
st.title("智能对话助手 🤖")
st.write("支持思考过程可视化的AI聊天机器人")

# 聊天记录管理
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=20)

# 分离实时显示和历史记录渲染
if "pending_response" not in st.session_state:
    st.session_state.pending_response = False


def render_historical_messages():
    """渲染历史消息（排除正在处理的最后一条用户消息）"""
    # 排除最后一条用户消息（正在处理中的消息）
    messages_to_render = list(st.session_state.chat_history)[
                         :-1] if st.session_state.pending_response else st.session_state.chat_history

    for chat in messages_to_render:
        with st.chat_message(chat["role"]):
            formatted = format_think_content(chat["content"])
            st.markdown(formatted, unsafe_allow_html=True)


def render_realtime_interaction():
    """渲染实时交互消息"""
    if st.session_state.pending_response:
        # 显示最后一条用户消息
        last_user_msg = st.session_state.chat_history[-1]
        with st.chat_message(last_user_msg["role"]):
            st.markdown(last_user_msg["content"], unsafe_allow_html=True)

        # 显示助理响应占位符
        with st.chat_message("assistant"):
            st.session_state.response_placeholder = st.empty()


# ============== 用户交互 ==============
with st.sidebar:
    model = st.selectbox("选择模型", ["deepseek-r1", "llama3", "mixtral"])
    st.button("清空记录", on_click=lambda: st.session_state.chat_history.clear())

prompt = st.chat_input("请输入您的问题...")

if prompt and not st.session_state.pending_response:
    # 标记开始处理新请求
    st.session_state.pending_response = True

    # 添加用户消息到历史记录
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 构建请求
    history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}"
         for msg in st.session_state.chat_history[:-1]]  # 排除当前正在处理的用户消息
    )

    try:
        # 发送请求
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"对话历史：\n{history}\nAssistant:",
                "stream": True
            },
            stream=True,
            timeout=30
        )
        response.raise_for_status()

        # 流式处理
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                if 'response' in data:
                    full_response += data['response']
                    # 更新实时显示
                    formatted = format_think_content(full_response + "▌")
                    st.session_state.response_placeholder.markdown(formatted, unsafe_allow_html=True)

        # 处理完成
        formatted_final = format_think_content(full_response)
        st.session_state.response_placeholder.markdown(formatted_final, unsafe_allow_html=True)

        # 添加助理响应到历史记录
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        # 启动语音线程
        threading.Thread(
            target=tts_speak,
            args=(full_response,),
            daemon=True
        ).start()

    except Exception as e:
        logging.error(f"请求失败: {str(e)}")
    finally:
        # 重置处理状态
        st.session_state.pending_response = False

# 页面渲染流程
render_historical_messages()  # 渲染历史消息
render_realtime_interaction()  # 渲染实时交互