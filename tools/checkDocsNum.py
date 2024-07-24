from llama_index.llms.qianfan import Qianfan
import asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader

# 定义全局参数
access_key = ""
secret_key = ""
endpoint_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"
model_name = "ERNIE-Speed-8K"
context_window = 8192

# 全局化定义嵌入层模型与大语言模型
Settings.embed_model = HuggingFaceEmbedding(model_name='D:/develop/pythonProject/llama/embedding-model/bge-base-zh-v1.5')
Settings.llm = Qianfan(access_key, secret_key, model_name, endpoint_url, context_window)

parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("../data", file_extractor=file_extractor).load_data()

print("文档总数：", len(documents))
print("第一个文档", documents[4])
print("第二个文档", documents[5])
print("倒数第二个文档", documents[6])
print("最后一个文档", documents[7])
# vector_index = VectorStoreIndex.from_documents(documents)
#
# # 创建查询器
# query_engine = vector_index.as_query_engine()   # 直接查询
#
# response = query_engine.query("《A LLM Benchmark based on the Minecraft Builder Dialog Agent Task》的作者是谁")
#
# print(response)