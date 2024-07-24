from llama_index.llms.qianfan import Qianfan
from llama_index.core.base.llms.types import ChatMessage

access_key = ""
secret_key = ""
endpoint_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"
model_name = "ERNIE-Speed-8K"
context_window = 8192

llm = Qianfan(access_key, secret_key, model_name, endpoint_url, context_window)

messages = [
    ChatMessage(role="user", content="《A LLM Benchmark based on the Minecraft Builder Dialog Agent Task》的作者是谁"),
]
response = llm.chat(messages)
print(response.message.content)
# content = ""
# for chat_response in llm.stream_chat(messages):
#     content += chat_response.delta
#     print(chat_response.delta, end="")
