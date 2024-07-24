import logging
import sys
from llama_index.llms.qianfan import Qianfan
from llama_index.core import PromptTemplate, Settings, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# 定义日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 定义全局参数
access_key = ""
secret_key = ""
endpoint_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"
model_name = "ERNIE-Speed-8K"
context_window = 8192

# 全局化定义嵌入层模型与大语言模型
Settings.embed_model = HuggingFaceEmbedding(model_name='D:/develop/pythonProject/llama/embedding-model/bge-base-zh-v1.5')
Settings.llm = Qianfan(access_key, secret_key, model_name, endpoint_url, context_window)

# 定义qa prompt
qa_prompt_tmpl_str = (
    "上下文信息如下。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请根据上下文信息而不是先验知识来回答以下的查询。"
    "作为一个文献内容检索智能助手，你的回答要尽可能严谨。\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# 定义refine prompt
refine_prompt_tmpl_str = (
    "原始查询如下：{query_str}"
    "我们提供了现有答案：{existing_answer}"
    "我们有机会通过下面的更多上下文来完善现有答案（仅在需要时）。"
    "------------"
    "{context_msg}"
    "------------"
    "考虑到新的上下文，优化原始答案以更好地回答查询。 如果上下文没有用，请返回原始答案。"
    "Refined Answer:"
)
refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)

# 使用LlamaDebugHandler构建事件回溯器，以追踪LlamaIndex执行过程中发生的事件
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

# 从存储文件中读取embedding向量和向量索引
storage_context = StorageContext.from_defaults(persist_dir="D:/develop/pythonProject/llama/vectorIndex")
index = load_index_from_storage(storage_context)

# 构建retriever
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# 构建synthesizer
response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.REFINE)

# 构建查询引擎
# query_engine = index.as_query_engine(similarity_top_k=3)
query_engine = RetrieverQueryEngine(retriever=retriever,
                                    response_synthesizer=response_synthesizer,
                                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)])
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl,
     "response_synthesizer:refine_template": refine_prompt_tmpl}
)
# 查询获得答案
response = query_engine.query("简要介绍在《Large Language Models through theLens of Students》中作者的发现")
print(response)

# get_llm_inputs_outputs返回每个LLM调用的开始/结束事件
event_pairs = llama_debug.get_llm_inputs_outputs()
# print(event_pairs[0][1].payload.keys())
