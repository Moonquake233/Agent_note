# Agent_note

记录我跟随 [Datawhale Hello-Agents《从零开始构建智能体》](https://github.com/datawhalechina/hello-agents) 学习 AI 智能体 (Agent) 的过程，包括代码练习、中文注释和基础模型实验。

这个仓库是个人学习记录。相关 Agent 示例参考 Hello-Agents 教程进行学习与实践；具体实现仍在逐步整理、调试和完善，不代表官方实现，也不保证所有脚本已经完整跑通。

## 学习内容

- **模型调用**：使用 OpenAI 兼容接口调用大语言模型 (LLM)，理解消息结构、环境变量配置和流式输出。
- **ReAct（推理与行动）**：练习模型生成行动、调用工具、接收观察结果并继续执行的循环。
- **Plan-and-Solve（规划与求解）**：先生成任务计划，再结合历史结果逐步执行。
- **Reflection（反思）**：学习执行记录与反馈的组织方式，目前只包含部分记忆模块。
- **基础补充**：练习 Transformer 组件，以及 Qwen 小模型的加载和文本生成。

## 目录说明

```text
Agent_note/
├── agent_1/
│   ├── client.py       # LLM 客户端封装与流式调用示例
│   ├── prompt.py       # ReAct、规划器与执行器的提示词模板
│   ├── Tool.py         # 工具注册、调用与 SerpApi 搜索
│   ├── React.py        # ReAct Agent 学习实现
│   ├── Plan_Solve.py   # 规划与逐步求解示例
│   └── Reflection.py   # 反思相关记忆模块，尚未完成
├── test.py             # 天气查询与景点推荐的旅行助手练习
├── qwen_0.5b.py         # Qwen1.5-0.5B-Chat 本地推理示例
├── transformer.py      # Transformer 结构与张量运算练习
├── 每天一次.py          # 基础组件重复练习草稿
├── .gitignore
└── README.md
```

`test.py` 是学习示例，不是自动化测试文件。`agent_1/.env` 为本地配置文件。


## 参考与致谢

主要学习来源：Datawhale 社区的 **Hello-Agents《从零开始构建智能体》**。感谢原项目作者与贡献者提供的教程和示例。

- [Hello-Agents 官方仓库](https://github.com/datawhalechina/hello-agents)
- [在线教程](https://datawhalechina.github.io/hello-agents/)
- [官方 PDF 发布页](https://github.com/datawhalechina/hello-agents/releases/latest)

引用、改编或传播源自教程的内容时，请保留来源说明并遵循原项目对应内容的许可要求。本仓库中的基础练习不统一声明为个人原创成果。
