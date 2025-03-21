import streamlit as st
import pyttsx3
import requests
import re
import json
from collections import deque
import threading
import logging
import os

# Streamlit 1.18+ 支持 add_script_run_ctx，用于多线程上下文传递
from streamlit.runtime.scriptrunner import add_script_run_ctx

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

# ============== 初始化 TTS 引擎 ==================
def init_tts_engine():
    """初始化 pyttsx3 语音引擎"""
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

# 若在 streamlit run 环境中，可以做一次初始化
if os.environ.get('STREAMLIT_RUNNING'):
    init_tts_engine()

# ============== 辅助函数 ==================
def format_think_content(text: str) -> str:
    """
    格式化 <think>...</think> 内容为 HTML 样式块
    在展示时将其变为灰色斜体块
    """
    return re.sub(
        r'<think>(.*?)</think>',
        r'<div class="think-text">\1</div>',
        text,
        flags=re.DOTALL
    )

def tts_speak(text: str):
    """语音播报：去除 <think> 内容后调用 pyttsx3"""
    if not text:
        return
    try:
        with _TTS_LOCK:
            if _TTS_ENGINE is None:
                init_tts_engine()
            if _TTS_ENGINE:
                # 去掉 <think> 思考标签只读可见内容
                cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
                _TTS_ENGINE.say(cleaned.strip())
                _TTS_ENGINE.runAndWait()
    except Exception as e:
        error_msg = f"语音错误: {str(e)}"
        logging.error(error_msg)
        with open(_TTS_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{error_msg}\n")

# ============== 页面结构 ==================
st.title("智能对话助手 🤖")
st.write("支持思考过程可视化的AI聊天机器人")

# 初始化 session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=20)  # 保存最近20条对话
if "processing" not in st.session_state:
    st.session_state.processing = {"active": False, "assistant_message": ""}

# 用于渲染所有对话的容器
chat_container = st.container()

def render_chat():
    """
    统一在同一个容器中渲染对话历史 + 当前流式回复
    """
    chat_container.empty()
    with chat_container:
        # 先渲染历史记录
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(format_think_content(item["content"]), unsafe_allow_html=True)
        # 若 AI 正在生成回复，则渲染一个“assistant”消息框显示流式文本
        if st.session_state.processing.get("active", False):
            with st.chat_message("assistant"):
                current_text = st.session_state.processing.get("assistant_message", "")
                # 结尾加个闪烁符号
                st.markdown(format_think_content(current_text + "▌"), unsafe_allow_html=True)

# ============== 侧边栏 ==================
with st.sidebar:
    model = st.selectbox("选择模型", ["deepseek-r1", "llama3", "mixtral"])
    if st.button("清空记录"):
        st.session_state.chat_history.clear()
        st.experimental_rerun()

# ============== 主体：用户输入 + 后台处理 ==================
user_input = st.chat_input("请输入您的问题...")

if user_input and not st.session_state.processing.get("active", False):
    # 1. 用户输入立即显示到对话记录
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    render_chat()  # 立刻更新界面，让用户消息可见

    # 2. 设置处理状态，准备获取 AI 回复
    st.session_state.processing = {"active": True, "assistant_message": ""}

    def get_ai_response():
        """
        在后台线程中流式获取 AI 回复，并实时更新界面。
        """
        try:
            # 构建历史上下文
            history_text = "\n".join(
                [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history]
            )
            # 调用你的后端接口，这里以 localhost:11434 为例
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": f"{history_text}\nAssistant:",
                    "stream": True
                },
                stream=True,
                timeout=30
            )
            response.raise_for_status()

            full_response = ""
            # 流式接收
            for line in response.iter_lines():
                if line:
                    # 调试：打印原始返回
                    print("DEBUG raw line:", line)
                    data = json.loads(line.decode())
                    print("DEBUG parsed data:", data)

                    # 如果你的服务端返回的字段不是 'response'，请改成正确字段名
                    if 'response' in data:
                        chunk = data['response']
                        full_response += chunk
                        st.session_state.processing["assistant_message"] = full_response
                        # 实时刷新
                        render_chat()

            # 3. 将最终回复加入历史
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            # 4. 启动语音播报（可选）
            threading.Thread(target=tts_speak, args=(full_response,), daemon=True).start()

        except Exception as e:
            logging.error(f"请求失败: {str(e)}")
        finally:
            # 5. 流式结束，重置处理状态
            st.session_state.processing = {"active": False, "assistant_message": ""}
            render_chat()

    # 创建后台线程并传递当前的 ScriptRunContext
    worker_thread = threading.Thread(target=get_ai_response, daemon=True)
    add_script_run_ctx(worker_thread)
    worker_thread.start()

#（可选）如果想在没有输入时也随时更新界面，可放开这行
# render_chat()
