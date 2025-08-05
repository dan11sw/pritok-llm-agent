import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("cckevinn/SeeClick", device_map="cuda", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

img_path = "test_img.png"
prompt = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"
# prompt = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with bbox)?"  # Use this prompt for generating bounding box
ref = "add an event"   # response (0.17,0.06)
ref = "switch to Year"   # response (0.59,0.06)
ref = "search for events"   # response (0.82,0.06)
query = tokenizer.from_list_format([
    {'image': img_path}, # Either a local path or an url
    {'text': prompt.format(ref)},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)



