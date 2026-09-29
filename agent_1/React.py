from prompt import REACT_PROMPT_TEMPLATE
from client import HelloAgentsLLM
from Tool import ToolExecutor,search
import re
class ReActAgent:
    def __init__(self,llm_client:HelloAgentsLLM,tool_executor:ToolExecutor,max_steps:int = 5):
        self.llm_client = llm_client
        self.tool_exectuor = tool_executor
        self.max_steps = max_steps
        self.history = []

            #解析LLM的输出
    def _parse_output(self,text:str):
        """
        Thought:       找到字面文本 Thought:
        \s一个空白字符，包括空格、制表符、换行
        * 前面的内容重复零次或多次
            \s*            跳过紧随其后的空白
        (.*?)          捕获内容，尽可能少地匹配       (.*)捕获内容，尽可能多匹配
        (...)捕获组：把匹配到的这部分保存下来用 .group(1) 取第一组
        $ 字符串末尾
        (?=...)：正向先行断言（Positive Lookahead）,检查后面是什么,但不把后面的内容纳入匹配。
        (?=\nAction:|$) 后面必须是“换行 + Action:”或者字符串结束
        re.DOTALL让 . 也能匹配换行
        """
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)",text,re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$",text,re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought,action
    
    def _parse_action(self,action_text:str):
        """ 解析action字符串，提取工具名称和输入
        (\w+)    第一组：匹配一个或多个单词字符，作为工具名
        \[       匹配字面符号 [
        (.*)     第二组：匹配括号中的内容，作为参数
        ]        匹配字面符号 ] 也可写作\]
        """
        match = re.match(r"(\w+)\[(.*)]",action_text,re.DOTALL)
        if match :
            return match.group(1),match.group(2)
        return None,None
    
    def run(self,question:str):
        self.history = [] #每次运行重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"-------第{current_step}步------")
            #格式化提示词
            tools_desc = self.tool_exectuor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools = tools_desc,
                question = question,
                history = history_str
            )
            #调用LLM进行思考
            message = [{"role":"user","content":prompt}]
            response_text = self.llm_client.think(messages=message)
            if not response_text:
                print("错误:LLM未能成功响应")
                break
            thought,action = self._parse_output(response_text)
            if thought:
                 print(f"思考:{thought}")
            if not action:
                 print("警告:未能解析出有效的Action,流程终止。")
                 break
            #执行action
            if action.startswith("Finish"):
                final_answer = re.match(r"Finish\[(.*)]",action).group(1)
                return final_answer

            tool_name,tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                 continue
            print(f"行动:{tool_name}[{tool_input}]")
            tool_function = self.tool_exectuor.getTool(tool_name)
            if not tool_function:
                # {} 外面的单引号：是文字，会输出。
                # {} 里面的 'name'：是 Python 字符串，用于取字典值，不会输出引号。
                observation = f"错误：未找到名为'{tool_name}'的工具"
            else:
                observation = tool_function(tool_input)
                print(f"观察{observation}")
                #将本轮的Action和obse添加到历史记录中
                self.history.append(f"Action:{action}")
                self.history.append(f"Observation:{observation}")
        print("已到达最大步数,流程终止")
        return  None

if __name__ =="__main__":
     llm = HelloAgentsLLM()
     tool_executor = ToolExecutor()
     search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
     tool_executor.regiserTool("Search",search_desc,search)
     agent = ReActAgent(llm_client=llm,tool_executor=tool_executor)
     question = "华为最新的手机是哪一款？它的主要卖点是什么？"
     agent.run(question=question)
