import os, sys
from typing import Any, Dict
from pydantic import BaseModel

class Config(BaseModel):
    # llm配置
    model: str = "qwen-plus-2025-12-01"
    provider: str = "qwen"
    temperature: float = 0.5
    max_tokens: int|None = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"


    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
     
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 测试默认值
    config = Config()
    print(config)
    print(config.to_dict())
    print()

    # 测试自定义值
    config = Config(model="gpt-4", temperature=0.8, max_tokens=1000)
    print(config)
    print(config.to_dict())
    print()


    # 测试 from_env
    os.environ["DEBUG"] = "true"
    os.environ["TEMPERATURE"] = "0.9"
    config = Config.from_env()
    print(config)
    print(config.to_dict())
    print()