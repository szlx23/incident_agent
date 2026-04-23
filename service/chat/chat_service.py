from datetime import datetime
from typing import AsyncIterator

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from service.rag.retrieval_service import RetrievalService

_retrieval_service = RetrievalService()


def build_system_prompt():
    from textwrap import dedent
    return dedent("""\
        你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。
        
        工作原则：
        1. 理解用户需求，选择合适的工具完成任务
        2. 当需要实时信息或专业知识时，主动使用相关工具
        3. 基于工具返回的结果提供准确、专业的回答
        4. 如果工具无法提供足够的信息，请如实告知用户

        回答要求：
        1. 保持友好、专业的语气
        2. 回答简洁明了、重点突出
        3. 基于事实，不编造信息
        4. 如有不确定的地方，请如实说明

        请根据用户的问题，灵活使用工具，提供高质量的帮助。"""
    ).strip()


class ChatService:
    def __init__(self, streaming: bool = True):
        from config import Config

        # create_agent 期望的是可调用的聊天模型实例，这里使用 OpenAI 兼容模型客户端封装。
        self.model = ChatOpenAI(
            model=Config.doubao_model_id,
            api_key=Config.ark_api_key,
            base_url=Config.doubao_base_url,
            streaming=streaming
        )
        self.tools = [retrieve_knowledge, get_current_time]
        self.checkpointer = MemorySaver()
        self.agent = None
        self._agent_initialized = False
        self.system_prompt = build_system_prompt()

    async def _initialize_agent(self):
        if self._agent_initialized:
            return

        all_tools = self.tools

        self.agent = create_agent(
            model=self.model,
            tools=all_tools,
            checkpointer=self.checkpointer
        )

        self._agent_initialized = True

    async def query(self, question: str, session_id: str) -> str:
        await self._initialize_agent()

        message = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question)
        ]

        result = await self.agent.ainvoke(
            input={"messages": message},
            config={"configurable": {"thread_id": session_id} }
        )

        return result["messages"][-1].content

    async def query_stream(self, question: str, session_id: str) -> AsyncIterator[str]:
        # 先确保 agent 已经完成初始化，避免首次调用时报错。
        await self._initialize_agent()

        # 组装输入消息：系统提示词 + 用户问题。
        message = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question)
        ]

        # 使用消息级流式输出，持续接收模型生成中的增量事件。
        async for chunk, _ in self.agent.astream(
            input={"messages": message},
            config={"configurable": {"thread_id": session_id}},
            stream_mode="messages"
        ):
            # 将 chunk 转成可直接展示的纯文本。
            text = _extract_chunk_text(chunk)
            # 忽略空片段，只向上游透传有效文本。
            if text:
                yield text



def retrieve_knowledge(query: str) -> str:
    """从知识库检索与问题相关的信息。"""
    return _retrieval_service.retrieve_knowledge(query=query, k=4)


def get_current_time() -> str:
    """返回当前系统时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _extract_chunk_text(chunk) -> str:
    # 从 chunk 对象里读取 content；没有就给空字符串。
    content = getattr(chunk, "content", "")
    # content 本身是字符串时，直接返回。
    if isinstance(content, str):
        return content

    # content 是列表时，逐项提取并拼接文本。
    if isinstance(content, list):
        parts = []
        for item in content:
            # dict 结构通常来自多模态/结构化消息片段。
            if isinstance(item, dict):
                # 标准文本片段：{"type": "text", "text": "..."}。
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
                # 兼容 {"content": "..."} 这类结构。
                elif item.get("content") and isinstance(item.get("content"), str):
                    parts.append(item["content"])
            # item 本身就是字符串时，直接收集。
            elif isinstance(item, str):
                parts.append(item)
        # 合并为一个连续字符串返回。
        return "".join(parts)

    # 其他未知结构统一降级为空字符串，避免抛异常打断流式响应。
    return ""

    
