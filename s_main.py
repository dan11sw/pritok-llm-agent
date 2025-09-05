from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nvmem
import sys, os, gc
import time

# Загрузка промпт-запроса
dir_path = os.path.abspath(os.path.dirname(__file__))
filename_prompt = 'prompt_3.txt'

settings = {
    "filepath": dir_path + os.path.sep + filename_prompt,
    "mode": "r"
}

prompt = ""
with open(settings["filepath"], settings["mode"]) as f:
    prompt = f.read()

torch.cuda.empty_cache()

nvmem.printInfoCUDA()
nvmem.printMemoryUsed()

#model_name = "microsoft/Phi-3-mini-4k-instruct"
#model_name = "ai-forever/rugpt3small_based_on_gpt2"
#model_name = "Qwen/Qwen2.5-Coder-3B"
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

tokenizer = AutoTokenizer.from_pretrained("/home/qwen2.5-coder")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device).eval()

nvmem.printMemoryUsed()

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

print("text: ", text)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

nvmem.printMemoryUsed()

start_time = time.time()
generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
        temperature=0.1
)
end_time = time.time()

print("LEN RESPONSE: ", len(generated_ids))

elapsed_time = end_time - start_time
print("Time: ", elapsed_time)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

nvmem.clearMemory()
nvmem.printMemoryUsed()

