from fastapi import FastAPI
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import torch
import uvicorn
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()

# tokenizer = AutoTokenizer.from_pretrained("viktoroo/sberbank-rubert-base-collection3")
# model = AutoModelForTokenClassification.from_pretrained("viktoroo/sberbank-rubert-base-collection3")

# @app.post("/get_prediction")
# async def get_prediction(item: Item):
#     try:
#         text = item.text
#         inputs = tokenizer(text, return_tensors="pt")
#
#         with torch.no_grad():
#             outputs = model(**inputs)
#
#         predicted_labels = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
#
#         token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
#
#         token_labels = [model.config.id2label[label_id] for label_id in predicted_labels]
#
#         print(token_list, token_labels)
#
#         return {"tokens": token_list, "labels": token_labels}
#     except Exception as e:
#         print(e)

@app.get("/get_prediction")
async def get_prediction(item: Item):
    try:
        text = item.text

        print(text)

        token_list = ['[CLS]', 'вице', '-', 'премьер', 'по', 'социальным', 'вопросам', 'та', '##тья', '##на', 'гол', '##ик', '##ова', 'рассказала', ',', 'в', 'каких', 'регионах', 'россии', 'зафиксирована', 'наиболее', 'высокая', 'смертность', 'от', 'рака', ',', 'сообщает', 'ри', '##а', 'новости', '.', 'по', 'словам', 'гол', '##ик', '##ово', '##и', ',', 'чаще', 'всего', 'онкологи', '##ческие', 'заболевания', 'становились', 'причин', '##ои', 'смерти', 'в', 'псков', '##ско', '##и', ',', 'твер', '##ско', '##и', ',', 'ту', '##льск', '##ои', 'и', 'орлов', '##ско', '##и', 'областях', ',', 'а', 'также', 'в', 'севасто', '##поле', '.', 'вице', '-', 'премьер', 'напомнила', ',', 'что', 'главные', 'факторы', 'смертности', 'в', 'россии', '—', 'рак', 'и', 'болезни', 'системы', 'кровообращения', '.', 'в', 'начале', 'года', 'стало', 'известно', ',', 'что', 'смертность', 'от', 'онкологических', 'заболевании', 'среди', 'россиян', 'снизилась', 'впервые', 'за', 'три', 'года', '.', 'по', 'данным', 'рос', '##стата', ',', 'в', '2017', 'году', 'от', 'рака', 'умерли', '289', 'тысяч', 'человек', '.', 'это', 'на', '3', ',', '5', 'процента', 'меньше', ',', 'чем', 'годом', 'ранее', '.', '[SEP]']
        token_labels = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'I-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        return {"tokens": token_list, "labels": token_labels, "text": text}
    except Exception as e:
        print(e)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8091)
