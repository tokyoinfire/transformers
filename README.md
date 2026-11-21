# Sentiment Analysis с BERT

## Описание
Проект по анализу тональности текстов с использованием трансформеров: недельный курс (токенизация →
hidden states → attention → baseline → fine-tuning → сравнение → анализ ошибок и демо) на отзывах IMDB
(бинарная классификация: 0 — negative, 1 — positive), модель `distilbert-base-uncased`.

## Структура
- `transformers.ipynb` — весь код курса, по дням (запускать сверху вниз)
- `data/imdb_sample.csv` — сбалансированная выборка IMDB (создаётся ноутбуком)
- `models/fine_tuned_model/` — обученная модель
- `models/baseline_model.pkl` — baseline (LogisticRegression на CLS-эмбеддингах)
- `app.py` — Gradio приложение
- `api.py` — FastAPI сервис
- `output/error_analysis.txt` — анализ ошибок
- `output/comparison_results.txt` — сравнение моделей
- `output/*.png` — графики attention и confusion matrix

## Запуск демо
Сначала выполните ноутбук целиком — он создаст `models/fine_tuned_model/`.

### Вариант с Gradio:
```bash
pip install gradio transformers torch
python app.py
```
Откройте http://127.0.0.1:7860

### Вариант с FastAPI:
```bash
pip install fastapi uvicorn transformers torch
uvicorn api:app --reload
```
API будет доступен на http://127.0.0.1:8000 (документация — `/docs`)

## Результаты
Заполняется последней ячейкой ноутбука.

<!-- results:start -->
### Fine-tuned модель:
- F1 (macro): 0.8884
- Accuracy: 0.8885

### Baseline модель:
- F1 (macro): 0.8271
- Accuracy: 0.8272

Улучшение: 7.41%
<!-- results:end -->

## Использование в коде
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('./models/fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./models/fine_tuned_model')

# Предсказание
inputs = tokenizer("Your text here", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1).item()
print(model.config.id2label[pred])
```

## Требования
- Python 3.10+
- transformers, torch
- pandas, scikit-learn, matplotlib, seaborn, joblib, datasets (для ноутбука)
- gradio или fastapi + uvicorn (для демо)
