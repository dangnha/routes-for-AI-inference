from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession, require_roles
from app.core.auth import AuthUser
from app.models import InferenceTask
from app.schemas import TaskCreate, TaskRead
from app.services.inference import dispatch_inference

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post("", response_model=TaskRead, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    payload: TaskCreate,
    background_tasks: BackgroundTasks,
    user: CurrentUser,
    db: DbSession,
) -> InferenceTask:
    task = InferenceTask(
        id=str(uuid4()),
        owner_id=user.id,
        filename=payload.filename or payload.object_key,
        object_key=payload.object_key,
        model=payload.model,
        status="pending",
        progress=0,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    background_tasks.add_task(
        dispatch_inference,
        task.id,
        task.object_key,
        task.model,
        task.filename,
    )
    return task


@router.get("", response_model=list[TaskRead])
def list_tasks(
    user: CurrentUser,
    db: DbSession,
    offset: int = 0,
    limit: int = 20,
) -> list[InferenceTask]:
    limit = min(max(limit, 1), 100)
    statement = (
        select(InferenceTask)
        .where(InferenceTask.owner_id == user.id)
        .order_by(InferenceTask.created_at.desc())
        .offset(max(offset, 0))
        .limit(limit)
    )
    return list(db.scalars(statement))


@router.get("/admin/all", response_model=list[TaskRead])
def list_all_tasks(
    _: Annotated[AuthUser, Depends(require_roles("admin"))],
    db: DbSession,
) -> list[InferenceTask]:
    statement = select(InferenceTask).order_by(InferenceTask.created_at.desc()).limit(100)
    return list(db.scalars(statement))


@router.get("/{task_id}", response_model=TaskRead)
def get_task(task_id: str, user: CurrentUser, db: DbSession) -> InferenceTask:
    task = db.get(InferenceTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.owner_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    return task
