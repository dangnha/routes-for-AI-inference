"""Request and response schemas, kept separate."""

from pydantic import BaseModel, ConfigDict, Field

# Request: what the client sends
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, le=20, description="Sepal length (cm)")
    sepal_width: float = Field(..., gt=0, le=20, description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, le=20, description="Petal length (cm)")
    petal_width: float = Field(..., gt=0, le=20, description="Petal width (cm)")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
            ]
        }
    )

    def to_vector(self) -> list[float]:
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


# Response: what we promise to return
class ProbabilityItem(BaseModel):
    label: str
    probability: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    model: str
    predicted_label: str
    predicted_class: int
    confidence: float = Field(..., ge=0, le=1)
    probabilities: list[ProbabilityItem]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
