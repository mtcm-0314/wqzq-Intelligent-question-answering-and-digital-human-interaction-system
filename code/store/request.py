import httpx
from nonebot.log import logger

from ..config import config
from ..compat import model_dump

# from ..function_call import registry
from ..exception import RequestException
from ..schemas import Balance, ChatCompletions
import json

# class API:
#     _headers = {
#         "Accept": "application/json",
#     }
    

#     @classmethod
#     async def chat(cls, message: list[dict[str, str]], model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") -> ChatCompletions:
#         """普通对话"""
#         model_config = config.get_model_config(model)

#         api_key = model_config.api_key or config.api_key
#         # api_key="sk-aamgdovwgwalykadxfwdkbipuusdggapytopbblgihybnakn"
#         prompt = model_dump(model_config, exclude_none=True).get("prompt", config.prompt)

#         json = {
#             "messages": [{"content": prompt, "role": "system"}] + message if prompt else message,
#             "model": model,
#             **model_config.to_dict(),
#         }
#         logger.debug(f"使用模型 {model}，配置：{json}")
#         # if model == "deepseek-chat":
#         #     json.update({"tools": registry.to_json()})
#         async with httpx.AsyncClient() as client:
#             response = await client.post(
#                 f"{model_config.base_url}/chat/completions",
#                 headers={**cls._headers, "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
#                 json=json,
#                 timeout=50,
#             )
#         if error := response.json().get("error"):
#             raise RequestException(error["message"])
#         return ChatCompletions(**response.json())

#     @classmethod
#     async def query_balance(cls, model_name: str) -> Balance:
#         model_config = config.get_model_config(model_name)
#         api_key = model_config.api_key or config.api_key

#         async with httpx.AsyncClient() as client:
#             response = await client.get(
#                 f"{model_config.base_url}/user/balance",
#                 headers={**cls._headers, "Authorization": f"Bearer {api_key}"},
#             )
#         if response.status_code == 404:
#             raise RequestException("本地模型不支持查询余额，请更换默认模型")
#         return Balance(**response.json())


class API:
    @classmethod
    async def chat(cls, message: list[dict[str, str]], model: str = "Pro/deepseek-ai/DeepSeek-V3") -> ChatCompletions:
        model_config = config.get_model_config(model)
        
        api_key = model_config.api_key or config.api_key

        # 修正base_url设置（原错误点1）
        base_url = model_config.base_url.rstrip("/")  # 确保没有结尾斜杠
        endpoint = f"{base_url}/chat/completions"
        prompt = model_config.prompt or config.prompt  # 获取配置中的prompt

        # # 构造完整messages
        # if model=="deepseek-ai/DeepSeek-V3":
        #     messages = [{"role": "system", "content": prompt}] + message if prompt else message
        # else:
        #     messages=message

        messages = [{"role": "system", "content": prompt}] + message if prompt else message
        json = {
            "messages": messages,  # 使用包含system消息的完整列表
            "model": model,
            **model_config.to_dict(),
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=json,
                timeout=50,
            )

        # 添加HTTP状态码检查（原错误点2）
        if response.status_code != 200:
            error_msg = f"API请求失败: {response.status_code}\n响应内容: {response.text}"
            raise RequestException(error_msg)

        try:
            response_data = response.json()
        except json.JSONDecodeError: # type: ignore
            raise RequestException(f"响应解析失败，原始内容: {response.text}")

        if "error" in response_data:
            raise RequestException(response_data["error"].get("message", "Unknown error"))
        
        return ChatCompletions(**response_data)

