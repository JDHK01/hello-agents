from typing import Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

MessageRole = Literal[
    "user", "assistant", "system", "tool"
]
    
class Message(BaseModel):
    '''
    消息类
    '''
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content
        }

if __name__ == "__main__":
    # 创建消息
    msg1 = Message(role="user", content="你好，请介绍一下自己")
    print(msg1)
    print(f"时间戳: {msg1.timestamp}")
    print(f"转为字典: {msg1.to_dict()}")

    # 带 metadata 的消息
    msg2 = Message(
        role="assistant",
        content="我是一个AI助手",
        metadata={"model": "gpt-4", "tokens": 50}
    )
    print(msg2)
    print(f"元数据: {msg2.metadata}")