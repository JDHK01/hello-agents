import sys, os
from .message import Message
from .config import Config
from .my_llm import MyLLM
from typing import Any
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(
        self, 
        llm: MyLLM,
        name: str|None = "anonymous",
        config: Config|None = None,
        system_prompt: str|None = None     
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()

        # 会话记录
        self._history: list[Message] = []

    
    @abstractmethod
    def run(self, input: str, **kwargs) -> str:
        pass


    def update_history(self, message:Message):
        self._history.append(message)

    
    def clear_history(self):
        self._history.clear()

    
    def get_history(self):
        return self._history.copy()
    

    def __str__(self):
        return f"agent: {self.name}"

