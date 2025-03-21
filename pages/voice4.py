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
    color: #888888;
    font-style: italic;
    border-left: 3px solid #e0e0e0;
    padding-left: 10px;
    margin: 5px 0;
    opacity: 0.7;
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

            # 测试语音功能
            _TTS_ENGINE.say("系统准备就绪")
            _TTS_ENGINE.runAndWait()
            _TTS_ENGINE.endLoop()
    except Exception as e:
        logging.error(f"语音初始化失败: {str(e)}")


if os.environ.get('STREAMLIT_RUNNING'):
    init_tts_engine()


# ============== 功能函数 ==============
def format_think_content(text):
    """将<think>标签内容转换为带样式的HTML"""
    return re.sub(
        r'<think>(.*?)</think>',
        r'<div class="think-text">\1</div>',
        text,
        flags=re.DOTALL
    )


def tts_speak(text):
    """语音播报（清理思考内容后）"""
    if not text:
        return

    try:
        with _TTS_LOCK:
            if _TTS_ENGINE is None:
                init_tts_engine()

            if _TTS_ENGINE:
                # 清理思考内容并播报
                cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                _TTS_ENGINE.say(cleaned.strip())
                _TTS_ENGINE.runAndWait()
    except Exception as e:
        logging.error(f"语音错误: {str(e)}")
        with open(_TTS_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{str(e)}\n")


# ============== 页面组件 ==============
st.title("智能对话助手 🤖")
st.write("对话流程：用户输入 → AI思考 → 正式回答 → 语音播报")

# 初始化状态管理
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=20)  # 存储已完成的对话对

if "processing" not in st.session_state:
    st.session_state.processing = {
        "user_input": None,  # 当前用户输入
        "ai_think": None,  # AI思考内容（带<think>标签）
        "ai_final": None,  # AI最终回答
        "placeholder": None  # 流式响应占位符
    }


def render_interface():
    """统一渲染界面"""
    # 渲染历史记录
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(format_think_content(chat["content"]), unsafe_allow_html=True)

    # 渲染当前处理中的交互
    if st.session_state.processing["user_input"]:
        # 用户消息（立即显示）
        with st.chat_message("user"):
            st.markdown(st.session_state.processing["user_input"])

        # AI思考过程+回答（流式更新）
        if st.session_state.processing["ai_think"]:
            with st.chat_message("assistant"):
                # 合并思考和最终回答
                full_content = st.session_state.processing["ai_think"]
                if st.session_state.processing["ai_final"]:
                    full_content += st.session_state.processing["ai_final"]

                # 实时更新显示
                if st.session_state.processing["placeholder"]:
                    formatted = format_think_content(full_content + "▌")
                    st.session_state.processing["placeholder"].markdown(formatted, unsafe_allow_html=True)
                else:
                    st.markdown(format_think_content(full_content), unsafe_allow_html=True)


# ============== 用户交互 ==============
with st.sidebar:
    model = st.selectbox("选择AI模型", ["deepseek-r1", "llama3", "mixtral"])
    st.button("清空记录", on_click=lambda: st.session_state.chat_history.clear())

prompt = st.chat_input("请输入您的问题...")

if prompt and not st.session_state.processing["user_input"]:
    # 初始化处理状态
    st.session_state.processing = {
        "user_input": prompt,
        "ai_think": "",
        "ai_final": "",
        "placeholder": None
    }

    try:
        # 立即渲染用户输入
        render_interface()

        # 构建请求上下文（仅使用历史记录）
        history = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}"
             for msg in st.session_state.chat_history]
        )

        # 发送请求
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"{history}\nUser: {prompt}\nAssistant:",
                "stream": True
            },
            stream=True,
            timeout=30
        )
        response.raise_for_status()

        # 初始化占位符
        with st.chat_message("assistant"):
            st.session_state.processing["placeholder"] = st.empty()

        # 流式处理响应
        buffer = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                if 'response' in data:
                    buffer += data['response']

                    # 分离思考内容和最终回答
                    think_match = re.search(r'<think>(.*?)</think>', buffer, re.DOTALL)
                    if think_match:
                        st.session_state.processing["ai_think"] = think_match.group(0)
                        st.session_state.processing["ai_final"] = re.sub(r'<think>.*?</think>', '', buffer)
                    else:
                        st.session_state.processing["ai_final"] = buffer

                    # 实时更新界面
                    render_interface()

        # 最终处理
        final_think = st.session_state.processing["ai_think"] or ""
        final_answer = st.session_state.processing["ai_final"] or ""

        # 提交到历史记录（严格用户-AI顺序）
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"{final_think}{final_answer}"
        })

        # 启动语音线程（仅播报最终回答）
        threading.Thread(
            target=tts_speak,
            args=(final_answer,),
            daemon=True
        ).start()

    except Exception as e:
        logging.error(f"请求失败: {str(e)}")
        # 清理无效记录
        if prompt in [msg["content"] for msg in st.session_state.chat_history]:
            st.session_state.chat_history.pop()
    finally:
        # 重置处理状态
        st.session_state.processing = {
            "user_input": None,
            "ai_think": None,
            "ai_final": None,
            "placeholder": None
        }

# 主渲染入口
render_interface()