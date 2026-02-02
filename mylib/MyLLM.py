import os, sys
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

class MyLLM:
    def __init__(self, model:str|None=None, api_key:str|None=None, base_url:str|None=None, timeout:int|None=None):
        self.model = model or os.getenv("DEFAULT_OPENAI_MODEL")
        self.client = OpenAI(
            api_key= api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout or int(os.getenv("DEFAULT_TIMEOUT", 60))
        )
    
    def generate(self, messages:List[Dict[str,str]], temperature:float=0):
        print(f"llm输入: \n{messages}\n")

        response = self.client.chat.completions.create(
            model = self.model,
            messages=messages,
            temperature=temperature,
        )
        output = response.choices[0].message.content
        print(f"llm输出: \n{output}\n")

        return output
    
if __name__ == "__main__":
    myllm = MyLLM()
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁"}
    ]
    myllm.generate(
        messages=message
    )