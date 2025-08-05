from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

modelName = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    modelName, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto"
)

processor = AutoProcessor.from_pretrained(modelName)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/test_img.png",
            },
            {"type": "text", "text": "Определи координаты значка захабренный, кнопку для перехода в раздел новости и кнопку показать. Ответ представь в формате (x; y) для каждого элемента веб-страницы"},
        ],
    }
]

text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)


