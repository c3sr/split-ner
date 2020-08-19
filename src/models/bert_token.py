import torch
from transformers import BertTokenizer, BertModel

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer("Hello, my dog is cute".split(" "), return_tensors="pt", is_pretokenized=True)
    labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs)
    loss, scores = outputs[:2]
    print("abc")
    # tokenizer("Jatin".split(" "), return_tensors="pt", is_pretokenized=True)["input_ids"][0].shape
