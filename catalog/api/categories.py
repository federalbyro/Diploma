from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.db.models import Category
from shared.db.postgres import get_session
from shared.services.embedding import upsert_category_embeddings
from shared.services.preprocessing import build_category_text

router = APIRouter(prefix="/categories", tags=["categories"])


class CategoryUpdateRequest(BaseModel):
    category_name_ru: str | None = None
    category_description_ru: str | None = None
    category_descriptions: list[str] | None = Field(default=None)


@router.patch("/{category_id}")
def patch_category(category_id: int, payload: CategoryUpdateRequest) -> dict:
    with get_session() as session:
        category = session.get(Category, category_id)

        if category is None:
            raise HTTPException(status_code=404, detail="Category not found")

        if payload.category_name_ru is not None:
            category.category_name_ru = payload.category_name_ru.strip()

        if payload.category_description_ru is not None:
            category.category_description_ru = payload.category_description_ru.strip()

        category.normalized_text_ru = build_category_text(
            category.category_name_ru,
            category.category_description_ru,
        )

        session.flush()

        category_payload = {
            "category_name_ru": category.category_name_ru,
            "category_description_ru": category.category_description_ru,
            "category_descriptions": payload.category_descriptions,
        }

        upsert_category_embeddings(
            [category],
            category_payloads=[category_payload],
        )

        return {
            "status": "ok",
            "category_id": category.category_id,
            "category_name_ru": category.category_name_ru,
            "descriptions_indexed": len(
                payload.category_descriptions or [category.category_description_ru]
            ),
        }