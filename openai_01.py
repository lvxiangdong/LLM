import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex, VectorStoreIndex
nest_asyncio.apply()
os.environ["OPENAI_API_KEY"] = "sk-QIwQGYONjWRZ9iDxvIpkD2ydp2Hslrxl3aoKb6SjC5ePVVYR"
os.environ["BASE_URL"] = "https://api.chatanywhere.tech/v1"

# documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()

from llama_index.core.node_parser import SentenceSplitter

splitter =SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key='sk-QIwQGYONjWRZ9iDxvIpkD2ydp2Hslrxl3aoKb6SjC5ePVVYR',
                      api_base='https://api.chatanywhere.tech', temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002",
                                       api_key='sk-QIwQGYONjWRZ9iDxvIpkD2ydp2Hslrxl3aoKb6SjC5ePVVYR',
                                       api_base='https://api.chatanywhere.tech/v1')

# summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
vector_query_engine = vector_index.as_query_engine()

from llama_index.core.tools import QueryEngineTool

# summary_tool = QueryEngineTool.from_defaults(query_engine=summary_query_engine,description=("Useful for summarization questions related to MetaGPT"))
vector_tool = QueryEngineTool.from_defaults(query_engine=vector_query_engine, description=( "Useful for retrieving specific context from the MetaGPT paper."))

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

query_engine = RouterQueryEngine(selector=LLMSingleSelector.from_defaults(),query_engine_tools=[vector_tool],verbose=True)

# response = query_engine.query("What is the summary of the document?")
# print(response)

response = query_engine.query("这篇论文的标题是什么")
print(str(response))















