import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = 'models/fine_tuned_model'

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model_ft.eval()


def predict_sentiment(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    label_map = model_ft.config.id2label

    result = f"Prediction: {label_map[pred]}\n\n"
    result += "Probabilities:\n"
    for i, prob in enumerate(probs[0]):
        result += f"{label_map[i]}: {prob*100:.2f}%\n"

    return result


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis с BERT",
    description="Введите текст на английском, и модель определит его тональность"
)

if __name__ == "__main__":
    demo.launch()
