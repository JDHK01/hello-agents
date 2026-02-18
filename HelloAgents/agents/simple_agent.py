import sys, os
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from HelloAgents.core.message import Message
from HelloAgents.core.config import Config
from HelloAgents.core.my_llm import MyLLM
from HelloAgents.core.agent import Agent
from HelloAgents.tools.builtin.search import search, SEARCH_DESCRIPTION
from HelloAgents.tools.registry import ToolRegistry
from typing import Dict, List, Any
import json



class SimpleAgent(Agent):
    def __init__(
        self,
        name: str|None ,
        llm: MyLLM,
        system_prompt: str|None = None,
        config: Config|None = None,
        # 新增工具调用功能
        enable_tool_calling: bool = False,
        tool_registry: ToolRegistry|None = None 
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config
        )
        self.enable_tool_calling = enable_tool_calling
        self.tool_registry = tool_registry


    def run(self, input:str, max_tool_iteration:int=10, **kwargs):
        print('='*20 + f"{self.name} processing start: {input}" + '='*20)

        # 从 system_prompt 和 self.history 以及 input 整理出完整的输入
        messages: List[Dict[str, Any]]= []
        messages.append(
            {"role": 'system', "content": self._get_format_system_prompt()}
        )
        for i in self._history:
            messages.append(
                {'role': i.role, 'content': i.content}
            )    
        messages.append(
            {'role': 'user', "content": input}
        )

        # 如果没有工具调用
        if not self.enable_tool_calling:
            response = self.llm.invoke(
                messages=messages
            )
            response_json = json.loads(response)
            # 更新 self._history
            self.update_history(Message(role="user", content=input))
            self.update_history(Message(role='assistant', content=response))
            print('='*20 + f"{self.name} processing end: {input}" + '='*20)
            return response_json.get('final', '')

        # 正常处理
        final_answer = self._run_with_tools(messages=messages, input=input, max_tool_iteration=max_tool_iteration)
        self.update_history(Message(role="user", content=input))
        self.update_history(Message(role='assistant', content=final_answer))
        return final_answer


    def _run_with_tools(self, input, messages, max_tool_iteration):
        '''
        模板如下:
        {
        "type": "tool_call" | "final",
        "tool_calls": [{"name": string, "args": object}],
        "final": string
        }
        '''
        for step in range(max_tool_iteration):
            # 生成回复, 并解析为json格式
            response = self.llm.invoke(
                messages=messages
            )
            response_json = json.loads(response)
            messages.append(
                {"role": "assistant", "content": response}
            )

            # 如果说不需要调用工具
            if response_json.get("type", '') == "final":
                return response_json['final']
            
            # 需要调用工具
            tool_result = []
            for call in response_json.get("tool_calls", ''):
                tool_name = call.get('name', '')
                tool_args = call.get('args', '')
                result = self._execute_tool(
                    tool_name=tool_name,
                    args=tool_args
                )
                tool_result.append(
                    {"name": tool_name, "result": result}
                )
            messages.append(
                {"role": "user", "content": json.dumps({"tool_result": tool_result}, ensure_ascii=False)}
            )

            # 如果即使到最大循环, 还是不行, 强制输出结果
            messages.append(
                {"role": "user", "content": "You have enough information now. Return a final answer with type=\"final\"."}
            )
            response = self.llm.invoke(messages=messages)    
            response_json = json.loads(response)
            if response_json['type'] == 'final':
                return response_json['final']
            else:
                return '-1'            
            

    def _execute_tool(self, tool_name: str, args: dict):
        tool = self.tool_registry.useTool(tool_name)
        # args 是一个字典，如 {"query": "..."}
        # 需要解包成关键字参数传递给工具函数
        result = tool(**args)
        return str(result)

    
    def _get_format_system_prompt(self) -> str:
        '''
        模板如下

        <You are a helpful AI assistant.>You can use external tools when needed.

        ## Available tools
        {tools}

        ## Output contract (STRICT)
        You MUST output exactly one JSON object and nothing else (no markdown, no extra text).

        Schema:
        {
        "type": "tool_call" | "final",
        "tool_calls": [{"name": string, "args": object}],
        "final": string
        }

        Rules:
        1) If you need a tool, set type="tool_call" and provide tool_calls.
        2) Call tools ONLY from the available tool list. Never invent tool names.
        3) Do NOT hallucinate tool outputs. Use tools to get results.
        4) After a tool_call, you will receive a user message containing JSON:
        {"tool_results":[{"name": "...", "result": "..."}]}
        Then decide the next action.
        5) If you can answer, set type="final" and fill "final".
        6) Output JSON ONLY.

        '''
        base = self.system_prompt or "You are a helpful AI assistant."

        # 无工具时，也要求 JSON 格式
        if not self.enable_tool_calling or not self.tool_registry:
            return (
                base
                + "\n\n## Output contract (STRICT)\n"
                + "Output ONLY a JSON object. No extra text, no explanation before or after.\n"
                + 'Format: {"type": "final", "final": "your answer"}'
            )

        tools_desc = self.tool_registry.introduceTool()
        if not tools_desc or tools_desc == "No tools available":
            return (
                base
                + "\n\n## Output contract (STRICT)\n"
                + "Output ONLY a JSON object. No extra text, no explanation before or after.\n"
                + 'Format: {"type": "final", "final": "your answer"}'
            )

        return (
            base
            + "\n\n"
            + "## Available tools\n"
            + tools_desc
            + "\n\n"
            + "## Output contract (STRICT)\n"
            + "You MUST output exactly one JSON object and nothing else (no markdown, no extra text).\n\n"
            + "Schema:\n"
            + "{\n"
            + '  "type": "tool_call" | "final",\n'
            + '  "tool_calls": [{"name": string, "args": object}],\n'
            + '  "final": string\n'
            + "}\n\n"
            + "Rules:\n"
            + '1) If you need a tool, set type="tool_call" and provide tool_calls.\n'
            + "2) Call tools ONLY from the available tool list. Never invent tool names.\n"
            + "3) Do NOT hallucinate tool outputs. Use tools to get results.\n"
            + '4) After a tool_call, you will receive a user message containing JSON:\n'
            + '   {"tool_results":[{"name":"...","result":"..."}]}\n'
            + "   Then decide the next action.\n"
            + '5) If you can answer, set type="final" and fill "final".\n'
            + "6) Output JSON ONLY.\n"
        )