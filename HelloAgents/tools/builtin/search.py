from serpapi import SerpApiClient
import os
from dotenv import load_dotenv

load_dotenv()

SEARCH_DESCRIPTION = "search(query: str): A web search engine tool based on SerpApi. It intelligently parses search results and prioritizes returning direct answers or knowledge graph information."
def search(query: str) -> str:
    """
    A web search engine tool based on SerpApi
    It intelligently parses search results and prioritizes returning direct answers or knowledge graph information.
    """
    print('-'*10 + f"executing search" + '-'*10)
    print(f"question: {query}")
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
            result = "\n".join(results["answer_box_list"])
            print(result)
            print('-'*10 + 'execution done' + '-'*10 + '\n')
            return result
        if "answer_box" in results and "answer" in results["answer_box"]:
            result = results["answer_box"]["answer"]
            print(result)
            print('-'*10 + 'execution done' + '-'*10 + '\n')
            return result
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            result = results["knowledge_graph"]["description"]
            print(result)
            print('-'*10 + 'execution done' + '-'*10 + '\n')
            return result
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            result = "\n\n".join(snippets)
            print(result)
            print('-'*10 + 'execution done' + '-'*10 + '\n')
            return result
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"