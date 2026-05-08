from fastapi import FastAPI

from catalog.api import router as catalog_router
from shared.core.config import get_settings

app = FastAPI(
    title="Catalog Service",
    description="Service for handling supplier catalogs and product categorization",
    version="1.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(catalog_router)

@app.on_event("startup")
async def startup():
    """Ensure all directories exist."""
    settings = get_settings()
    settings.ensure_dirs()