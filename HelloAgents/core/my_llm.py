import sys, os

from .hello_agents_llm import HelloAgentsLLM
from openai import OpenAI

# from hello_agents import HelloAgentsLLM

from typing import Literal
SUPPORTED_PROVIDER = Literal[
    "qwen", "modelscope"
]


class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        provider: str|None = None,
        api_key: str|None = None,
        base_url: str|None = None,
        model: str|None = None,
        # **kwargs
    ):
        if provider == "modelscope":
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or os.getenv("MODELSCOPE_BASE_URL")
            self.model = model or os.getenv("MODELSCOPE_DEFAULT_MODEL")

            if not all(
                [self.api_key, self.base_url, self.model]
            ):
                raise ValueError("base_url | api_key | model is not defined")
            
            # 其他参数
            # self.temperature = kwargs.get("temperature", 0)
            # self.max_tokens = kwargs.get("max_tokens", 1000000)
            # self.timeout = kwargs.get("timeout", 300)

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                # timeout=self.timeout
            )

        elif provider == "qwen":
            self.base_url = base_url or os.getenv("QWEN_BASE_URL")
            self.api_key = api_key or os.getenv("QWEN_API_KEY")
            self.model = model or os.getenv("QWEN_DEFAULT_MODEL")

            if not all(
                [self.base_url, self.api_key, self.model]
            ):
                raise ValueError("base_url | api_key | model is not defined")
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

        else:
            super().__init__(
                model=model,
                api_key=api_key,
                base_url=base_url,
                # provider = provide
            )

if __name__ == "__main __":
    my_llm = MyLLM(
        provider="modelscope"
    )
    input = "who are you"
    messages = [
        {'role': "user", "content": input}
    ]

    my_llm.generate(
        messages=messages,

    )