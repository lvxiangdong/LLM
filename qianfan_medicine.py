from llama_index.llms.qianfan import Qianfan
import asyncio
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import CSVReader

# 定义全局参数
access_key = ""
secret_key = ""
endpoint_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"
model_name = "ERNIE-Speed-8K"
context_window = 8192

# 全局化定义嵌入层模型与大语言模型
Settings.embed_model = HuggingFaceEmbedding(model_name='D:/develop/pythonProject/llama/embedding-model/bge-base-zh-v1.5')
Settings.llm = Qianfan(access_key, secret_key, model_name, endpoint_url, context_window)

# 读取文件
# parser = CSVReader()
# file_extractor = {".csv": parser}
# 创建索引与嵌入向量
documents = SimpleDirectoryReader("./data").load_data()
vector_index = VectorStoreIndex.from_documents(documents)

# 持久化存储向量
vector_index.storage_context.persist("D:/develop/pythonProject/llama/vectorIndex")

# 加载持久化向量
# storage_vector = StorageContext.from_defaults(persist_dir="D:/develop/pythonProject/llama/vectorIndex")
# load_vector = load_index_from_storage(storage_vector)

# 创建查询器
# query_engine = vector_index.as_query_engine()   # 直接查询
# query_engine = load_vector.as_query_engine()    # 从本地库获取

# messages = [
#     ChatMessage(role="user", content="你知道GPT4吗"),
# ]
# # response = llm.chat(messages)
# # print(response.message.content)
# content = ""
# for chat_response in llm.stream_chat(messages):
#     content += chat_response.delta
#     print(chat_response.delta, end="")

# response = query_engine.query("《A LLM Benchmark based on the Minecraft Builder Dialog Agent Task》的作者是谁")
#
# print(response)