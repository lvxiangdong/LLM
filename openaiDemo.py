import logging
import sys
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import ChatMessage

# 记录日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# SYSTEM_PROMPT = """You are a helpful AI assistant."""
# query_wrapper_prompt = PromptTemplate(
#     "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
# )
#
# llm = HuggingFaceLLM(
#     context_window=4096,
#     max_new_tokens=2048,
#     generate_kwargs={"temperature": 0.0, "do_sample": False},
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name='/yldm0226/models/Qwen1.5-14B-Chat',
#     model_name='/yldm0226/models/Qwen1.5-14B-Chat',
#     device_map="auto",
#     model_kwargs={"torch_dtype": torch.float16},
# )
Settings.llm = OpenAI(api_key='',
                      api_base='https://api.chatanywhere.tech/v1',
                      temperature=0.2)
Settings.embed_model = OpenAIEmbedding(api_key='',
                                       api_base='https://api.chatanywhere.tech/v1')


documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

print(response)