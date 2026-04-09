from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from chat_parser import parse_chat
from compressor import compress_chunks
from formatter import format_context_pack

router = APIRouter()


class CompressRequest(BaseModel):
    chat_text: str
    model: Optional[str] = "gemini-2.0-flash"
    api_key: Optional[str] = None


@router.post("")
async def compress(req: CompressRequest):
    if not req.chat_text.strip():
        raise HTTPException(status_code=400, detail="chat_text cannot be empty")

    chunks = parse_chat(req.chat_text)
    structured = await compress_chunks(chunks, req.model, req.api_key)
    context_pack = format_context_pack(structured)

    return {
        "structured": structured,
        "context_pack": context_pack,
        "stats": {
            "input_chars": len(req.chat_text),
            "output_chars": len(context_pack),
            "compression_ratio": round(len(req.chat_text) / max(len(context_pack), 1), 2),
            "chunks_processed": len(chunks),
        }
    }
