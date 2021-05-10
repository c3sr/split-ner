import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

batch_sentences = ["Hello I'm a single sentence",
                    "another sentence",
                    "And the very very last one"]
batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(batch)

batch_of_second_sentences = ["I'm a sentence that goes with the first sentence",
                              "And I should be encoded with the second sentence",
                              "And I go with the very last one"]

batch = tokenizer(batch_sentences, batch_of_second_sentences, padding=True, truncation=True, return_tensors="pt")

print(batch)
