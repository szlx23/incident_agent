import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from service.chat.chat_service import ChatService

router = APIRouter()
# 复用一个 ChatService 实例，避免每次请求重复初始化模型与工具。
chat_service = ChatService(streaming=True)


class ChatQueryRequest(BaseModel):
    # 用户输入的问题，最小长度 1，避免空请求。
    question: str = Field(..., min_length=1, description="用户提问")
    # 会话 ID 可选；不传则服务端自动生成，便于多轮对话隔离。
    session_id: str | None = Field(default=None, description="会话ID，可选")


@router.post("/chat/query/stream")
async def query_stream(payload: ChatQueryRequest):
    # 二次兜底校验：拦截仅包含空白字符的 question。
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question 不能为空")

    # 未传 session_id 时自动生成，确保每次请求有可追踪的会话标识。
    session_id = payload.session_id or str(uuid4())

    # SSE 事件生成器：逐段产出模型流式结果。
    async def event_generator():
        try:
            # 持续消费服务层的流式输出，并转换为 SSE 的 data 事件。
            async for chunk in chat_service.query_stream(
                question=payload.question,
                session_id=session_id
            ):
                data = {"type": "chunk", "session_id": session_id, "content": chunk}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            # 全部输出完成后，发送 done 事件通知前端收流。
            done = {"type": "done", "session_id": session_id}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 异常时返回 error 事件，前端可统一处理错误展示。
            error = {"type": "error", "session_id": session_id, "message": str(e)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    # 使用标准 SSE 响应类型，浏览器/EventSource 可直接消费。
    return StreamingResponse(event_generator(), media_type="text/event-stream")
