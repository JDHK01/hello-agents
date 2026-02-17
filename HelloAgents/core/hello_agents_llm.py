from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class HelloAgentsLLM:
    '''
    基于 `base_url`, `api_key`, `model`调用模型, 默认参数为 .env 中的参数
    '''
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ):
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_DEFAULT_MODEL")

        # 检验是否存在参数错误
        if not all(
            [base_url, api_key]
        ):
            raise ValueError("base_url | api_key | model is not defined")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )


    def generate(self, messages, temperature:int=0):
        print('-'*10 + "llm invoking start" + '-'*10)
        print('messages:')
        print(messages)
        response = self.client.chat.completions.create(
            model = self.model,
            temperature = temperature,
            messages = messages
        )
        output = response.choices[0].message.content
        print('output:')
        print(output)
        print('-'*10 + "llm invoking over" + '-'*10)
        print()
        return output


    def invoke(self, messages, temperature:int=0):
        return self.generate(messages=messages, temperature=temperature)

    






if __name__ == "__main__":
    my_llm = HelloAgentsLLM()
    input = "hello"
    messages = [
        {"role": "user", "content": input}
    ]
    my_llm.generate(
        messages=messages
    )

    my_llm.invoke(
        messages=messages
    )