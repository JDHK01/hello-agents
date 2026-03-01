在保持外部接口不变的情况下, 我重写了 `hello-agent` 的代码(跳过了transformer部分). 并按照功能为模块放置到`mycode/`, 而不是按照章节, 分布更加科学

代码风格
- 注重实现, 只加入少量异常处理机制(仍然保证极高的成功率)
- 尽量充分的展现llm的处理过程, 将llm所有的输出输出全部打印, 方便调测.
- 部分细节改用更加科学的处理方式, 比如以基座为qwen的embedding模型, Template模块等
- 完备的测试机制, 每部分后面都会紧跟着一个测试模块, 

代码说明
- `helloworld.ipynb`: 一个简单的agent入门级实现
- `llm_basics.ipynb`: 大模型相关的基础知识, 比如分词, 解码等等
- `ReAct.ipynb`: reason-action范式
- `PlanAndSolve.ipynb`: plan-solve范式
- `reflection.ipynb`: reflection范式

# chapter1

- 使用强制llm输出`json`替代`re`, 更加现代化, 同步修改系统提示词
有效的避免了
```bash
output 
Thought: 已获取北京当前天气为晴天，气温7摄氏度。接下来我将调用get_attraction工具，根据城市“北京”和天气“sunny”推荐一个合适的旅游景点。
Action: get_attraction(city="北京", weather="sunny")
Observation: 推荐景点：颐和园。理由：晴天适合户外游览，颐和园风景优美，历史悠久，是北京经典景点之一。
Thought: 已获得推荐景点为颐和园，并有合理解释。现在可以向用户提供完整答复。
Action: Finish[今天北京天气晴朗，气温7摄氏度，适合户外活动。推荐您游览颐和园，这里风景优美、历史悠久，是晴天出游的理想选择。]
```
- 那个查询天气的api服务有问题, 经常网络错误, 我这里强制赋值, 不影响功能的体现

# chapter3

- embedding方式更加先进, 使用qwen的基座大模型生成向量而不是预定义的向量
- 针对mac做适配, 增加`mps`选项

# chapter4
PS: 预先说一下, 这里的`Tool.py`模块其实应该是`Action`模块, 但是主包懒的改了(不影响使用, 你该放工具放入工具就行)
- 类型注释推荐使用更现代化的, 防止报错
```python
class MyLLM:
    def __init__(self, model: str | None = None):
        self.model = model

from typing import Optional

class MyLLM:
    def __init__(self, model: Optional[str] = None):
        self.model = model

```

- 学习阶段, 不实用流式输出, 继续采用整体输出
- 应该把工具名name放到description里面, 不然类型注释无法组合
- 将终止融入到 action, 也就是让tool工具中默认包含finish. 这样就可以省掉一个json的键值对
- 依旧用json而不是re
- 全新的提示词
- 你用string默认的`string.format()`与json的`{}`互相排斥, 所以我该用了`from string import Template`, 达到同样的效果

# my_agent
PS: 这里就不严格按照官方的说明了
tip
- `messages`, `input`, `output`, `response`: input、output为从人类视角的输入输出, messages和response是从网络侧截取的内容, 除input和output外, 还包含一定的调试信息

`core/message.py`: 取消<pydantic.__init__>方法的调用, 该用 Field(default_factory=...)

.dict()已经被弃用, 现在是使用 .model_dump

<MyLLM>.provider为冗余参数, 这里省去

使用json代替re

实现最小demo

加入了notes作为说明字段

可以加入json修复repair, 暂未加入

ReActAgent没有刻意的记录一些内容(没有系统提示词, 一套提示词通吃)