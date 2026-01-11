from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from src.utils import load_trained_model, predict_digit, preprocess_image
from config import settings

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For now, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conv, pool, softmax, metadata = load_trained_model(settings.MODEL_PATH)


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "CNN Digit Recognition API",
        "model_info": metadata
    }


@app.get("/model/info")
def get_model_info():
    """Get information about the trained model"""
    return {
        "metadata": metadata,
        "architecture": {
            "conv_filters": conv.filters.shape,
            "pool_size": pool.pool_size,
            "softmax_weights": softmax.weights.shape,
            "softmax_biases": softmax.biases.shape
        }
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image file

    Accepts: PNG, JPG, or any common image format
    Returns: Predicted digit, confidence, and probability distribution
    """

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img_array = preprocess_image(img)

        digit, confidence, probabilities = predict_digit(img_array, conv, pool, softmax)

        return {
            "digit": int(digit),
            "confidence": float(confidence),
            "probabilities": {
                str(i): float(prob) for i, prob in enumerate(probabilities)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict/base64")
async def predict_base64(data: dict):
    """
    Predict digit from base64 encoded image

    Request body: {"image": "base64_string"}
    Returns: Predicted digit, confidence, and probability distribution
    """
    import base64

    if "image" not in data:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request body")

    try:
        # Decode and preprocess base64 image
        image_data = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(image_data))
        img_array = preprocess_image(img)

        digit, confidence, probabilities = predict_digit(img_array, conv, pool, softmax)

        return {
            "digit": int(digit),
            "confidence": float(confidence),
            "probabilities": {
                str(i): float(prob) for i, prob in enumerate(probabilities)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


