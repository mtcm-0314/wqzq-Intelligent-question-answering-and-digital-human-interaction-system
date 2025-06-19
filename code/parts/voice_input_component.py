import streamlit as st
import streamlit.components.v1 as components


def voice_input_component():
    st.markdown("### 🎙️ 点击按钮进行语音输入")

    components.html(
        """
        <button id="recordButton" style="padding:10px 20px;font-size:18px;">🎤 开始语音输入</button>
        <p id="transcript" style="margin-top:15px;"></p>

        <script>
            const button = document.getElementById("recordButton");
            const transcriptDisplay = document.getElementById("transcript");
            let recognizing = false;
            let recognition;

            if (!('webkitSpeechRecognition' in window)) {
                transcriptDisplay.innerHTML = "❌ 当前浏览器不支持语音识别。";
            } else {
                recognition = new webkitSpeechRecognition();
                recognition.lang = "zh-CN";
                recognition.interimResults = false;
                recognition.continuous = false;

                recognition.onresult = function(event) {
                    const result = event.results[0][0].transcript;
                    transcriptDisplay.innerHTML = "✅ 识别结果：" + result;

                    // 发送结果到父窗口
                    window.parent.postMessage({ 
                        type: 'voice_result', 
                        text: result 
                    }, "*");
                };

                recognition.onerror = function(event) {
                    transcriptDisplay.innerHTML = "❌ 识别失败: " + event.error;
                };

                recognition.onend = function() {
                    recognizing = false;
                    button.innerText = "🎤 开始语音输入";
                };

                button.onclick = () => {
                    if (!recognizing) {
                        recognition.start();
                        recognizing = true;
                        button.innerText = "🛑 停止语音输入";
                    } else {
                        recognition.stop();
                        recognizing = false;
                        button.innerText = "🎤 开始语音输入";
                    }
                };
            }
        </script>
        """,
        height=250
    )