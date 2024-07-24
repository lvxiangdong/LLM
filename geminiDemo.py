import os
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'

GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


messages = [
    ChatMessage(role="user", content="你好!"),
    ChatMessage(role="assistant", content="我有什么能帮助你"),
    ChatMessage(
        role="user", content="帮我决定一下今天的晚餐。"
    ),
]

# resp = Gemini(transport='rest', model="models/gemini-1.5-flash").complete("介绍武汉大学")
resp = Gemini(transport='rest', model="models/gemini-1.5-pro").chat(messages)
print(resp)