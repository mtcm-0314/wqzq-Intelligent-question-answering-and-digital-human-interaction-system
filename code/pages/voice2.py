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
        # 修复1：写入文件时指定utf-8编码
        with open(_TTS_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{error_msg}\n")


# ============== 页面组件 ==============
st.title("智能对话助手 🤖")
st.write("支持思考过程可视化的AI聊天机器人")

# 错误显示
if os.path.exists(_TTS_ERROR_LOG):
    try:
        # 修复2：读取时指定utf-8编码
        with open(_TTS_ERROR_LOG, "r", encoding="utf-8") as f:
            errors = f.readlines()
            if errors:
                st.error(f"最近语音错误: {errors[-1].strip()}")
    except UnicodeDecodeError as e:
        st.error(f"日志解码失败，请检查文件编码: {str(e)}")
    except Exception as e:
        st.error(f"日志读取失败: {str(e)}")

# 聊天记录管理
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=20)


def render_chat():
    """渲染聊天记录（带样式格式化）"""
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            formatted = format_think_content(chat["content"])
            st.markdown(formatted, unsafe_allow_html=True)


render_chat()

# ============== 用户交互 ==============
with st.sidebar:
    model = st.selectbox("选择模型", ["deepseek-r1", "llama3", "mixtral"])
    st.button("清空记录", on_click=lambda: st.session_state.chat_history.clear())

prompt = st.chat_input("请输入您的问题...")
if prompt:
    # 添加用户消息
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 构建请求
    history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}"
         for msg in st.session_state.chat_history]
    )

    # 处理响应
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
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
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    if 'response' in data:
                        full_response += data['response']
                        # 实时显示带样式的文本
                        formatted = format_think_content(full_response + "▌")
                        placeholder.markdown(formatted, unsafe_allow_html=True)

            # 最终显示
            formatted_final = format_think_content(full_response)
            placeholder.markdown(formatted_final, unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            # 启动语音线程
            threading.Thread(
                target=tts_speak,
                args=(full_response,),
                daemon=True
            ).start()

        except Exception as e:
            st.error(f"请求失败: {str(e)}")