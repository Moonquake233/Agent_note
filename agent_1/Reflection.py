from typing import List,Dict,Any,Optional
class Memory:
    def __init__(self):
        #初始化一个空列表存储所有记录
        self.records:List[Dict[str,Any]] =[]
    def add_record(self,record_type:str,content:str):  
        """
        向记忆中添加一条新记录。
        输入的是 一条记录的类型 和其对应的上下文
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        record = {
            "tpye":record_type,
            "content":content
        }
        self.records.append(record)
        print(f"记忆更新，新增一条'{record_type}'记录")

    def get_trajectory(self) -> str:
        #讲所有记忆格式化为连贯的字符串，用于构建提示词。
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(f"---上一轮尝试(代码)---\n{record['content']}")
            elif record['tpye'] == 'reflection':
                trajectory_parts.append(f"---评审员反馈-----\n{record['content']}")
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:  #Optional===Union[X, None]  这个值的 “类型” 要么是 X，要么是 None。
        #获取最近一次的执行结果 (例如：最新生成的代码)。
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None
