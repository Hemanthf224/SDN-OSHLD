from fastapi import APIRouter, Depends, HTTPException
from .schemas import InferenceRequest, InferenceResponse
from .inference import get_inference_service, InferenceService

router = APIRouter()

@router.post("/infer", response_model=InferenceResponse)
async def infer_flow(request: InferenceRequest):
    try:
        service = get_inference_service()
        flow_dict = request.features.model_dump()
        result = service.process_flow(flow_dict)
        return InferenceResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "SDN-OSHLD API"}
