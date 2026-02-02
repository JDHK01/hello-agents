from serpapi import SerpApiClient
import os
from dotenv import load_dotenv
from typing import Dict, List, Any


load_dotenv()


class ToolExecutor:
    '''
    工具管理器, 默认包含终止行动(旨在将finish变为action); 所以这里的ToolExecutor其实是ActionExecutor
    '''
    def __init__(self):
        self.tools : Dict[str, Dict[str, Any]] = {}
        self.registerTool(
            name = "finish",
            description="finish(conclusion: str): Call this action when you think it's time to end, args is your conclusion",
            function=None
        )


    def registerTool(self, name:str, description:str, function:Any):
        self.tools[name] = {
            "description": description,
            "function":function
        }


    def useTool(self, name):
        return self.tools.get(name).get("function")
    

    def introduceTool(self):
        return '\n'.join(
            [
                f'- {info.get("description")}' for info in self.tools.values()
            ]
        )


SEARCH_DESCRIPTION = "search(query: str): A web search engine tool based on SerpApi. It intelligently parses search results and prioritizes returning direct answers or knowledge graph information."
def search(query: str) -> str:
    """
    A web search engine tool based on SerpApi
    It intelligently parses search results and prioritizes returning direct answers or knowledge graph information.
    """
    print(f"正在执行[SerpApi]网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            # "gl": "cn",
            # "hl": "zh-cn",
            "gl": "us",      # 国家代码
            "hl": "en",      # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


if __name__ =="__main__":
    # test search
    query = "Tell me the latest price of gold"
    response = search(query=query)
    print(response)

    # test ToolExecutor
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        name="search",
        description=SEARCH_DESCRIPTION,
        function=search
    )
    print(tool_executor.introduceTool())
    special_tool = tool_executor.useTool(
        name="search"
    )
    observation = special_tool(query=query)
    print(observation)