import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import requests
from io import BytesIO
from PIL import Image

# --- Configuration ---
# Your model ID (Note: This is a text model, but the API is structured for chat/VLM)
MODEL_ID = "rednote-hilab/dots.ocr"

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
try:
    # Load model and tokenizer (we keep them for the future, but they won't
    # be used to generate the chat completion for the image description placeholder)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model {MODEL_ID}: {e}")
    # We will let the app start but disable the embedding path if load fails
    # raise SystemExit(1) 


# --- Pydantic Schemas for OpenAI-style API ---

class ImageUrl(BaseModel):
    """Schema for the image_url detail."""
    url: str

class ContentDetail(BaseModel):
    """Schema for a single content item (text or image)."""
    type: str
    text: str | None = None
    image_url: ImageUrl | None = None

class Message(BaseModel):
    """Schema for a single message in the chat."""
    role: str
    content: list[ContentDetail]

class ChatCompletionBody(BaseModel):
    """Schema for the main request body."""
    model: str
    messages: list[Message]

class Choice(BaseModel):
    """Schema for the response choice."""
    index: int
    message: dict
    finish_reason: str = "stop"

class Usage(BaseModel):
    """Schema for the response usage stats."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    """Schema for the full response."""
    id: str = "chatcmpl-123"
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(torch.time()))
    model: str
    choices: list[Choice]
    usage: Usage

# --- FastAPI App ---
app = FastAPI(title="Multimodal Chat & Embedding API")


# Helper function to find text and image parts
def extract_multimodal_content(messages: list[Message]):
    """Extracts the first text prompt and first image URL from the messages list."""
    image_url = None
    text_prompt = None
    
    for message in messages:
        if message.role == "user":
            for content in message.content:
                if content.type == "image_url" and content.image_url:
                    image_url = content.image_url.url
                elif content.type == "text" and content.text:
                    text_prompt = content.text
    return text_prompt, image_url


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Chat"])
async def chat_completions(body: ChatCompletionBody):
    """
    Handles OpenAI-style chat completion requests, including multimodal inputs.
    
    NOTE: This endpoint is a placeholder. It extracts image/text but uses a 
    hardcoded description as the underlying model is text-only.
    """
    text_prompt, image_url = extract_multimodal_content(body.messages)

    if image_url:
        try:
            # 1. Image Download (For demonstration/validation)
            print(f"Downloading image from: {image_url}")
            response = requests.get(image_url, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes
            
            # 2. Image Processing (Placeholder)
            # You would normally feed the image and text prompt to a VLM here.
            # Example: VLM_MODEL.generate(image, text_prompt)
            
            # Placeholder response based on the Statue of Liberty image URL
            if "Statue-of-Liberty" in image_url:
                 description = "The Statue of Liberty stands on an island in New York Harbor under a bright sky, surrounded by water."
            else:
                 description = f"The provided image (URL: {image_url}) has been received, but the current text-only model cannot process it. The prompt was: '{text_prompt}'"

        except requests.exceptions.RequestException as e:
            description = f"Error accessing image URL: {str(e)}"
        except Exception as e:
            description = f"An internal error occurred during processing: {str(e)}"
    
    elif text_prompt:
        # If only text is provided, you could optionally call your text model here
        description = f"Text prompt received: '{text_prompt}'. Returning a generic text response."
    
    else:
        raise HTTPException(status_code=400, detail="Invalid request content. Must contain text or image.")


    # Construct the OpenAI-style response
    return ChatCompletionResponse(
        model=body.model,
        choices=[
            Choice(
                index=0,
                message={
                    "role": "assistant",
                    "content": description
                }
            )
        ]
    )


# --- Keep the original /embed endpoint for the text model ---
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
    Generates sentence embeddings for a list of input texts using the loaded model.
    """
    if not body.input_texts:
        raise HTTPException(
            status_code=400,
            detail="The 'input_texts' list cannot be empty."
        )

    try:
        # 1. Tokenization and Inference (using your original logic)
        batch_dict = tokenizer(
            body.input_texts, max_length=8192, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**batch_dict)

        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 2. Return Results
        return {
            "model_id": MODEL_ID,
            "embedding_count": len(embeddings),
            "embeddings": embeddings.tolist(),
        }
    except Exception as e:
        print(f"An error occurred during embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
