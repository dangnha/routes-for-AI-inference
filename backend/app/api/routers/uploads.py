from fastapi import APIRouter

from app.api.deps import CurrentUser
from app.schemas import PresignRequest, PresignResponse
from app.services.storage import create_presigned_upload, object_key_for

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("/presign", response_model=PresignResponse)
def presign_upload(payload: PresignRequest, _: CurrentUser) -> PresignResponse:
    object_key = object_key_for(payload.filename)
    upload_url = create_presigned_upload(object_key, payload.content_type)
    return PresignResponse(object_key=object_key, upload_url=upload_url)
