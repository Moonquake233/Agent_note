from client import HelloAgentsLLM
from prompt import PLANNER_PROMPT_TEMPLATE,EXECUTOR_PROMPT_TEMPLATE

import ast
class Planner:
    def __init__(self,llm_client):
        self.llm_client = llm_client
    def plan(self,question:str) ->list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question = question)
        messages =[{"role":"user","content":prompt}]
        print("正在生成计划")
        response_text = self.llm_client.think(messages = messages)
        print(f"计划已生成:\n{response_text}")
        try:
            """
            response_text
                ↓ split("'''python")
                切开
                ↓ [1]
                取后半段
                ↓ split("'''")
                再切
                ↓ [0]
                取前半段
                ↓ strip()
                去首尾空白
            """
            plan_str = response_text.split("'''python")[1].split("'''")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)
            # isinstance(plan,list) 判断plan是不是列表
            return plan if isinstance(plan,list) else []
            # ValueError：值不合法   SyntaxError：语法不合法  IndexError：索引越界
        except (ValueError,SyntaxError,IndexError) as e:
            print(f"解析计划时出错: {e}")
            return []
        except Exception as e:
            return []
class Executor:
    def __init__(self,llm_client):
        self.llm_client = llm_client

    def execute(self,question:str,plan:list[str]) -> str:
        """
        根据计划，逐步执行并解决问题
        """
        history = ""
        print("\n-----正在执行计划------")
        for i,step in enumerate(plan):
            print(f"\n->正在执行步骤 {i+1}/{len(plan)}: {step}")
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question = question,
                plan = plan,
                history = history if history else "无",
                current_step = step
            )
            messages = [{"role":"user","content":prompt}]
            response_text = self.llm_client.think(messages = messages) or ""
            history += f"步骤 {i+1}: {step}\n结果:{response_text}\n\n"
            print(f"步骤 {i+1}已完成,结果:{response_text}")

        final_answer = response_text
        return final_answer

class PlanAndSolveAgent:
    def __init__(self,llm_client):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self,question:str):
        plan = self.planner.plan(question)
        if not plan:
            print("任务终止，无法生成有效行动计划")
            return 
        final_answer = self.executor.execute(question,plan)
        print(f"\n---任务完成---\n最终答案:{final_answer}")

if __name__ == '__main__':
    try:
        llm_client = HelloAgentsLLM()
        agent = PlanAndSolveAgent(llm_client)
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        agent.run(question)
    except ValueError as e:
        print(e)