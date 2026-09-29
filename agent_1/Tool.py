from serpapi import SerpApiClient
import os
from dotenv import load_dotenv
from typing import Dict,Any
load_dotenv()

def search(query:str) -> str:
    #基于SerpApi的实战网页搜索引擎
    print(f"正在执行[SerpApi]网页搜索:{query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "API未在.env中配置"
        params = {
            "engine": "google",
            "q": query,
            "api_key":api_key,
            "gl":"cn",
            "hl":"zh-cn",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            #如果没有直接答案，返回前三个结果的摘要
            snippets = [  # enumerate(...)会给每个元素附带一个序号   #res.get('title','') 有title 就取 title，没有就返回空字符串
                f"[{i+1}] {res.get('title','')}\n{res.get('snippet','')}" for i,res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
    except Exception as e:
        return f"搜索时发生错误：{e}"


class ToolExecutor:
    def __init__(self):
        self.tools:Dict[str,Dict[str,Any]] = {}

    def regiserTool(self,name: str,description:str,func:callable):
        # 向工具箱注册一个新工具。
        if name in self.tools:
            print(f"警告:工具 '{name}'已存在。")
        self.tools[name] = {"description":description,"func":func}
        print(f"工具{'name'}已注册")

    def getTool(self,name:str) -> callable:
        #根据名称获取一个工具的执行函数
        return self.tools.get(name,{}).get("func")

    def getAvailableTools(self) ->str:
        #获取所有可用工具的格式化描述
        return "\n".join(
            [
                f"-{name}:{info['description']}"
                for name,info in self.tools.items()
            ]
        )

if __name__ =="__main__":
    ToolExecutor = ToolExecutor()
    search_descrpiton = "一个网页搜索引擎。"
    ToolExecutor.regiserTool("Search",search_descrpiton,search)

    print("\n-----可用工具-----")
    print(ToolExecutor.getAvailableTools())

    tool_name = "Search"
    tool_input = "英伟达最新GPU型号是什么"

    tool_function = ToolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为{tool_name}的工具")
