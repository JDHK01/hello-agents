REACT_PROMPT = """You are an AI assistant that can use tools.

## Available Tools
{tools}

## Output Contract (STRICT)
You MUST output exactly ONE JSON object and NOTHING else (no markdown, no extra text).

Schema:
{{
  "type": "tool_call" | "final",
  "tool": {{
    "name": string,
    "args": object
  }} | null,
  "final": string | null,
  "notes": string | null
}}

Rules:
1) Execute ONLY ONE step at a time.
2) If you need a tool, set type="tool_call" and fill tool.name + tool.args. Set final=null.
3) If you can answer, set type="final" and fill final. Set tool=null.
4) Use ONLY tools from the Available Tools list. Never invent tool names.
5) notes is optional and must be brief (1 sentence). Do NOT provide step-by-step reasoning.

## Current Question
{question}

## Execution History (JSON)
{history}
"""
import os, sys
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from HelloAgents.core import Agent, Config, Message, MyLLM
from HelloAgents.tools.registry import ToolRegistry
from HelloAgents.tools.builtin.search import SEARCH_DESCRIPTION, search
from typing import List, Dict, Any
import json

class ReActAgent(Agent):
    def __init__(
        self,
        llm: MyLLM,
        name: str = "anonymous",
        config: Config|None = None,
        system_prompt: str|None = None,
        # tool
        tool_registry: ToolRegistry|None = None,
        max_iterations: int = 10,
    ):
        super().__init__(
            llm=llm,
            config=config,
            name=name
        )
        self.prompt_template = REACT_PROMPT
        # 新增工具调用
        self.tool_registry: ToolRegistry|None = tool_registry
        # 新增最大循环
        self.max_iterations = max_iterations
        # reson action特有的轨迹记录
        self.trajectory: List[Dict[str, Any]] = []

    
    def run(self, input:str, **kwargs):
        print('='*20 + f"{self.name} processing start: {input}" + '='*20)
        tool_description = '' # 无意义, 只是为了兼容 tool_registry 为 None 的情况
        if self.tool_registry is not None:
            tool_description = self.tool_registry.introduceTool()
        for step in range(self.max_iterations):
            print('-'*10 + f"step: {step}" + '-'*10)
            # 填充提示词
            prompt = self.prompt_template.format(
                tools = tool_description,
                question = input,
                history = json.dumps(self.trajectory, ensure_ascii=False, indent=2)
            )

            # 整理messages
            messages = []
            if self.system_prompt is not None:
                messages.append({'role':'system', 'content': self.system_prompt})
            messages.append({'role': 'user', 'content':prompt})

            # 响应
            response = self.llm.invoke(
                messages=messages
            )
            response_json = json.loads(response)
            '''
            Schema:
                {{
                "type": "tool_call" | "final",
                "tool": {{
                    "name": string,
                    "args": object
                }} | null,
                "final": string | null,
                "notes": string | null
                }}
            '''

            # 如果是 final
            if response_json['type'] == 'final':
                final_result = response_json['final']
                self.update_history(Message(role='user', content=input))
                self.update_history(Message(role='assistant', content=final_result))
                return final_result
            
            tool_name = response_json['tool'].get('name')
            tool_args = response_json['tool'].get('args')

            tool = self.tool_registry.useTool(name=tool_name)
            obs = tool(**tool_args)

            self.trajectory.append(
                {
                    "step": step+1,
                    "note": response_json['notes'],
                    "tool": {'name': tool_name, "args": tool_args},
                    "observation": obs
                }
            )
        final_answer = "Sorry, I cannot complete this task within the limited number of steps."
        self.update_history(Message(role='user', content=input))
        self.update_history(Message(role='assistant', content=final_answer))
        return final_answer