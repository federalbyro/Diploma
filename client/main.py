from fastapi import FastAPI

from client.api.search import router as client_router
from shared.core.config import get_settings

app = FastAPI(
    title="Client Service",
    description="Service for handling client queries and product searches",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add router for client query handling
app.include_router(client_router)

@app.on_event("startup")
async def startup():
    """Ensure all directories exist."""
    settings = get_settings()
    settings.ensure_dirs()