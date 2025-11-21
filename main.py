import torch
from fastapi import (
    FastAPI,
    HTTPException,
)
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel

# --- Configuration ---
# Change to your model id
MODEL_ID = "rednote-hilab/dots.ocr"

# Determine device (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading (Happens once on startup) ---
try:
    # Load tokenizer and model, and move the model to the determined device
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    model.eval()  # Set the model to evaluation mode
except Exception as e:
    # Exit if model loading fails
    print(f"Error loading model {MODEL_ID}: {e}")
    raise SystemExit(1)

# --- FastAPI App and Schemas ---
app = FastAPI(title="Text Embedding API", description=f"Using model: {MODEL_ID}")


class MainBody(BaseModel):
    """
    Request body schema for the embedding endpoint.
    """
    input_texts: list[str]


@app.post("/embed", tags=["Embeddings"])
async def create_embeddings(
    body: MainBody,
):
    """
    Generates sentence embeddings for a list of input texts.
    """
    if not body.input_texts:
        raise HTTPException(
            status_code=400,
            detail="The 'input_texts' list cannot be empty."
        )

    try:
        # 1. Tokenization
        batch_dict = tokenizer(
            body.input_texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)  # Move input tensors to the same device as the model

        # 2. Inference
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**batch_dict)

        # 3. Extract and Pool Embeddings
        # Get the embedding for the [CLS] token (typically the first token)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # 4. L2 Normalization
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # 5. Return Results
        # Convert the tensor to a list of lists before returning
        return {
            "model_id": MODEL_ID,
            "embedding_count": len(embeddings),
            "embeddings": embeddings.tolist(),
        }
    except Exception as e:
        print(f"An error occurred during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
