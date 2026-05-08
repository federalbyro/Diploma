from fastapi import APIRouter

from catalog.api.catalog import router as catalog_router
from catalog.api.categories import router as categories_router

router = APIRouter()
router.include_router(catalog_router)
router.include_router(categories_router)