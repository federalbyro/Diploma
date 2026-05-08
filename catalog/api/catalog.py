from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from catalog.tasks.celery_app import process_catalog_file
from shared.core.config import get_settings
from shared.db.models import Job, JobStatus
from shared.db.postgres import get_session

router = APIRouter(prefix="/catalog", tags=["catalog"])
settings = get_settings()


class JobResponse(BaseModel):
    job_id: int
    status: str
    filename: str


@router.post("/upload", response_model=JobResponse)
def upload_catalog(
    file: UploadFile = File(...),
    supplier_name: str = Form(...),
    meta_json: str | None = Form(default=None),
) -> JobResponse:
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    save_path = upload_dir / file.filename
    save_path.write_bytes(file.file.read())

    with get_session() as session:
        job = Job(
            filename=file.filename,
            status=JobStatus.pending.value,
        )
        session.add(job)
        session.flush()
        job_id = job.job_id

    process_catalog_file.delay(
        job_id=job_id,
        file_path=str(save_path),
        supplier_name=supplier_name,
        meta_json=meta_json,
    )

    return JobResponse(
        job_id=job_id,
        status=JobStatus.pending.value,
        filename=file.filename,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: int) -> JobResponse:
    with get_session() as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobResponse(
            job_id=job.job_id,
            status=job.status,
            filename=job.filename or "",
        )