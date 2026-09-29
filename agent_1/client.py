import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List,Dict

load_dotenv()

class HelloAgentsLLM:
    def __init__(self,model:str = None,apiKey:str = None,baseUrl:str = None,timeout:int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT",60))
        if not all([self.model,apiKey,baseUrl]):            #all([...])列表里的所有元素都必须为 True。
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")
        self.client = OpenAI(api_key=apiKey,base_url=baseUrl,timeout=timeout)

    def think(self,messages:List[Dict[str,str]],temperature:float = 0) -> str:

        print(f"正在调用{self.model}模型")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream = True,  #流式输出，因此要for循环读取
            )
            print("大模型响应成功:")
            #处理流式输出
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""  #如果content是None，则取""
                print(content,end="",flush=True)   #end默认是\n，所以要让end=""  flush：立刻让输出刷新到前端
                collected_content.append(content)
            print() #流式输出后换行
            return "".join(collected_content) #用空字符串 "" 作为连接符，把列表里的所有字符串按顺序拼起来。
        except Exception as e:
            print(f"调用模型API时发生错误:{e}")
            return None

if __name__ =='__main__':
    try:
        llmClient = HelloAgentsLLM()

        Messages = [
            {"role":"system","content":"You are a helpful assistant that writes Python code."},
            {"role":"user","content":"写一个快速排序算法"}
        ]
        print("---调用LLM---")
        responseText = llmClient.think(Messages)
        if responseText:
            print("\n\n--- 完整模型相应 ---")
            print(responseText)
    except ValueError as e:
        print(e)