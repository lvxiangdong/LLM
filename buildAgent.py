import nest_asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.qianfan import Qianfan
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from pathlib import Path
from tools.utils import get_doc_tools
nest_asyncio.apply()

# 定义全局参数
access_key = ""
secret_key = ""
endpoint_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"
model_name = "ERNIE-Speed-8K"
context_window = 8192
# 全局化定义嵌入层模型与大语言模型
Settings.embed_model = HuggingFaceEmbedding(model_name='D:/develop/pythonProject/llama/embedding-model/bge-base-zh-v1.5')
Settings.llm = Qianfan(access_key, secret_key, model_name, endpoint_url, context_window)

# 文件列表
data_directory = Path("data")
papers = [
    "A LLM Benchmark based on the Minecraft Builder Dialog Agent Task.pdf",
    "PATCH-LEVEL TRAINING FOR LARGE LANGUAGE MODELS.pdf",
    "The Future of Learning Large Language Models through the Lens of Students.pdf",
]

# 创建查询工具，对每个文档构建向量索引与总结索引
paper_to_tools_dict = {}
for paper in papers:
    print(f"getting tools for: {paper}")
    pdf_path = data_directory / paper
    vector_tool, summary_tool = get_doc_tools(str(pdf_path), pdf_path.stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

agent_worker = ReActAgent.from_tools(initial_tools)
response = agent_worker.query("《A LLM Benchmark based on the Minecraft Builder Dialog Agent Task》中作出了哪些贡献")
print(response)
# vector_query_engine = VectorStoreIndex.from_documents(documents, use_async=True).as_query_engine()
# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=vector_query_engine,
#         metadata=ToolMetadata(
#             name="documents",
#             description="PDF papers",
#         ),
#     ),
# ]
#
# query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools, use_async=True)
#
# response = query_engine.query(
#     "《A LLM Benchmark based on the Minecraft Builder Dialog Agent Task》有哪些参考文献"
# )
# print(response)
