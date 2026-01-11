import os
from typing import List

class Settings:
    APP_TITLE: str = "CNN Digit Recognition API"
    APP_DESCRIPTION: str = "API for predicting handwritten digits using a custom CNN"
    APP_VERSION: str = "1.0.0"

    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    MODEL_PATH: str = "trained_cnn_model.npz"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = ENVIRONMENT == "development"


settings = Settings()
