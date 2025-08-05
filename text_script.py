import json

data = {"hello": "world"}

with open("/home/pritok_llm/data/sft_train.json", "r") as json_file:
    values = json.load(json_file)
    print(len(values))



