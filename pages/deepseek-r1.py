import streamlit as st
import pyttsx3
import requests
import re
import json
import threading
from requests.exceptions import RequestException

# 初始化语音引擎
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
except Exception as e:
    st.error(f"语音引擎初始化失败: {str(e)}")

# DeepSeek API 配置（使用官方推荐配置）
DEEPSEEK_API_URL = "https://api.siliconflow.cn/v1/chat/completions"  # 官方正式地址
API_KEY = "sk-"  # 替换zen的密钥

# 模型配置（根据最新文档）
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 推荐使用官方标准模型标识

st.title("DeepSeek Chatbot 🤖")
st.write("Powered by DeepSeek R1")

# 初始化对话历史（保留最多20轮）
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 添加系统提示词（可选）
    st.session_state.messages.append({
        "role": "system",
        "content": "你是一个乐于助人的AI助手，用中文简洁清晰地回答问题"
    })

# 渲染历史消息（过滤系统消息）
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 用户输入处理
user_input = st.chat_input("请输入你的问题...")

if user_input:
    # 添加用户消息（保留原始输入）
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 构造API请求
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        "messages": st.session_state.messages[-20:],  # 保留最近20轮对话
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1024
    }
    response = requests.request("POST", DEEPSEEK_API_URL, json=payload, headers=headers)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        error_occurred = False

        try:
            # 带重试机制的请求（生产环境建议使用retrying库）
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(10, 30),  # 连接超时10秒，读取超时30秒
                verify=True  # 保持SSL验证
            )

            # 检查HTTP状态码
            if response.status_code != 200:
                st.error(f"API请求失败，状态码：{response.status_code}")
                try:
                    error_data = response.json()
                    st.error(f"错误详情：{error_data.get('message', '未知错误')}")
                except:
                    st.error(f"响应内容：{response.text[:200]}")
                error_occurred = True
                response.close()  # 显式关闭连接

            # 处理流式数据
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        decoded_chunk = chunk.decode('utf-8').strip()

                        # 过滤掉 'data: [DONE]'，防止 JSON 解析错误
                        if decoded_chunk == "data: [DONE]":
                            break

                        if decoded_chunk.startswith("data: "):
                            json_data = json.loads(decoded_chunk[6:])  # 去掉 'data: ' 前缀

                            # 检查finish_reason
                            if json_data["choices"][0]["finish_reason"] not in [None, "stop"]:
                                st.warning(f"生成中断原因：{json_data['choices'][0]['finish_reason']}")
                                break

                            # 获取内容增量
                            delta = json_data["choices"][0].get("delta", {})
                            chunk_content = delta.get("content", "")

                            if chunk_content:
                                full_response += chunk_content
                                message_placeholder.markdown(full_response + "▌")

                    except json.JSONDecodeError:
                        st.warning("收到非JSON格式数据: " + decoded_chunk[:100])
                    except KeyError as e:
                        st.warning(f"响应字段缺失: {str(e)}")

            # 最终处理
            if not error_occurred:
                # 清理响应内容
                cleaned_response = re.sub(r'<think>.*?</think>', '', full_response)

                # 更新界面和历史记录
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

                # 语音播报（使用多线程避免阻塞主线程）
                def speak():
                    try:
                        engine.say(cleaned_response)
                        engine.runAndWait()
                    except Exception as tts_error:
                        st.error(f"语音合成失败: {str(tts_error)}")

                # 启动一个新线程进行语音合成
                threading.Thread(target=speak).start()

                # 控制历史记录长度
                if len(st.session_state.messages) > 20:
                    st.session_state.messages = st.session_state.messages[-20:]

        except requests.exceptions.SSLError as e:
            st.error(f"SSL连接错误: {str(e)}\n建议检查：\n1. 系统证书是否更新\n2. 网络代理设置")
        except requests.exceptions.Timeout:
            st.error("请求超时，请检查网络连接或稍后重试")
        except requests.exceptions.ConnectionError:
            st.error("连接失败，请检查网络连接")
        except RequestException as e:
            st.error(f"请求异常: {str(e)}")
        except Exception as e:
            st.error(f"未知错误: {str(e)}")

