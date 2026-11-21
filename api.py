import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = 'models/fine_tuned_model'

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model_ft.eval()

app = FastAPI()


class PredictionRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    label_map = model_ft.config.id2label

    return {
        "prediction": label_map[pred],
        "probabilities": {
            label_map[i]: float(probs[0][i])
            for i in range(len(probs[0]))
        }
    }
