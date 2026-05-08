from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from client.services.query import handle_query

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    user_id: int | str | None = None


@router.post("")
def search(payload: SearchRequest) -> dict:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    return {
        "query": payload.query,
        "results": handle_query(
            query_text=payload.query,
            user_id=payload.user_id,
        ),
    }