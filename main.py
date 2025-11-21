from fastapi import (
    FastAPI,
    HTTPException,
    Body,
)
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel

# Change to your model id
model_id = "rednote-hilab/dots.ocr"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

app = FastAPI()


class MainBody(BaseModel):
    input_texts: list[str]


@app.post("/")
async def root(
    body: MainBody,
):
    try:
        batch_dict = tokenizer(  # Update according to your model requirements
            body.input_texts,
            max_length=8192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0][:768]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return {  # Update response structure if needed
            "embeddings": embeddings.tolist(),
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
