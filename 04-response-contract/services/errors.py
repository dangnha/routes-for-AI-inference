"""Domain errors - raised by the service, mapped to HTTP status codes in main.py."""


class LowConfidenceError(Exception):
    """Prediction confidence is below the requested threshold -> 400."""

    def __init__(self, confidence: float, threshold: float):
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(f"Confidence {confidence:.4f} is below the required threshold {threshold:.4f}")


class ModelNotFoundError(Exception):
    """Model id does not exist -> 404."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' was not found")
