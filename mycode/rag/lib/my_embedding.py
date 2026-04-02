from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 这里不调api了, 直接跑一个本地模型
# 本机跑不动
# class MyEmbedding():
#     def __init__(
#         self,
#         model_name: str = "BAAI/bge-m3"
#     ):
#         self.model = HuggingFaceEmbedding(
#             model_name=model_name,
#             trust_remote_code=True,
#             cache_folder='./model_cache'
#         )
    
#     def encode(self, text):
#         print(f'embedding input: {text}')
#         result = self.model._get_text_embedding(text)
#         print(f'embedding output: {result}')
#         return result
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os
import numpy as np

class MyEmbedding():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("EMBED_API_KEY"),
            base_url=os.getenv("EMBED_BASE_URL")
        )

    def embedding(self, inputs, dimensions:int=3072):
        print(f"embedding输入: \n{inputs}\n")
        response = self.client.embeddings.create(
            model=os.getenv("EMBED_MODEL_NAME"),
            input=inputs,
            dimensions=dimensions,
            encoding_format="float"
        )
        
        # 提取向量并转为矩阵
        outputs = np.array([data.embedding for data in response.data])
        print(f"embedding输出: \n{outputs}\n")
        return outputs


if __name__ == '__main__':
    my_embedding = MyEmbedding()
    # 支持批量输入
    input = [
        'who are you',
        'I',
        'am',
        '666'
    ]
    result = my_embedding.embedding(
        input
    )