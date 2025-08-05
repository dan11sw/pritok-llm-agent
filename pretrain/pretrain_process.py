import json
from process_utils import is_english_simple, bbox_2_point, bbox_2_bbox
import task_prompts
from tqdm import tqdm
import os
import random

web_imgs_path = "/home/pritok_llm/dataset/cpfs01/user/chengkanzhi/seeclick_web_imgs"
web_json_path = "/home/pritok_llm/dataset/seeclick_web.json"

web_train = []
with open(web_json_path, "r") as json_file:
    web_train = json.load(json_file)

print("Size download dataset: ", len(web_train))

web_loca_point = []
web_loca_bbox = []
web_ocr_point = []
web_ocr_bbox = []

num_ele_valid = 0

print("Processing web dataset")

for i, item in tqdm(enumerate(web_train)):
    img_filename = item["img_filename"]
    img_path = os.path.join(web_imgs_path, img_filename)

    eles_valid = []
    for ele in item["elements"]:
        if len([item for item in ele["bbox"] if item < 0]) != 0:
            continue
        if len(ele["instruction"]) > 60 or ele["instruction"].strip() == '':
            continue
        if ('{' in ele["instruction"]) or ('}' in ele["instruction"]):
            continue
        if not is_english_simple(ele["instruction"]):
            continue
        eles_valid.append(ele)

    if len(eles_valid) == 0:
        continue

    num_ele_valid += len(eles_valid)

    # text to point
    random.shuffle(eles_valid)
    conversations = []

    prompt = random.choice(task_prompts.web_loca_all_point_prompt)
    prompt += ' '

    for j, item in enumerate(eles_valid):
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}

            conv_user["value"] += prompt
            conv_user["value"] += item["instruction"]
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += item["instruction"]

        click_point = bbox_2_point(item["bbox"])
        conv_ai = {"from": "assistant", "value": click_point}

        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_loca_point.append({"id": "loca_point_{}".format(i), "conversations": conversations})

    # text to bbox
    random.shuffle(eles_valid)
    conversations = []

    prompt = random.choice(task_prompts.web_loca_all_bbox_prompt)
    prompt += ' '

    for j, item in enumerate(eles_valid):
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            
            conv_user["value"] += prompt
            conv_user["value"] += item["instruction"]
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += item["instruction"]

        click_point = bbox_2_bbox(item["bbox"])
        conv_ai = {"from": "assistant", "value": click_point}

        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_loca_bbox.append({"id": "loca_bbox_{}".format(i), "conversations": conversations})

    # point to text
    random.shuffle(eles_valid)
    conversations = []

    prompt = random.choice(task_prompts.web_ocr_all_point_prompt)
    prompt += ' '

    for j, item in enumerate(eles_valid):
        click_point = bbox_2_point(item["bbox"])

        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
           
            conv_user["value"] += prompt
            conv_user["value"] += click_point
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += click_point

        conv_ai = {"from": "assistant", "value": item["instruction"]}

        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_ocr_point.append({"id": "ocr_point_{}".format(i), "conversations": conversations})

    # bbox to text
    random.shuffle(eles_valid)
    conversations = []

    prompt = random.choice(task_prompts.web_ocr_all_bbox_prompt)
    prompt += ' '

    for j, item in enumerate(eles_valid):
        click_point = bbox_2_bbox(item["bbox"])

        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += click_point
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += click_point

        conv_ai = {"from": "assistant", "value": item["instruction"]}

        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_ocr_bbox.append({"id": "ocr_bbox_{}".format(i), "conversations": conversations})

print("Num of valid elements: " + str(num_ele_valid))
print("Num of web_loca_point: " + str(len(web_loca_point)))
print("Num of web_loca_bbox: " + str(len(web_loca_bbox)))
print("Num of web_ocr_point: " + str(len(web_ocr_point)))
print("Num of web_ocr_bbox: " + str(len(web_ocr_bbox)))

random.shuffle(web_loca_point)
web_loca_point = web_loca_point[:]

random.shuffle(web_loca_bbox)
web_loca_bbox = web_loca_bbox[:54000]

random.shuffle(web_ocr_point)
web_ocr_point = web_ocr_point[:54000]

random.shuffle(web_ocr_bbox)
web_ocr_bbox = web_ocr_bbox[:54000]

sft_train = web_loca_point + web_loca_bbox + web_ocr_point + web_ocr_bbox

print("Num of sft: " + str(len(sft_train)))

with open("/home/pritok_llm/data/sft_train.json", "w") as json_file:
    json.dump(sft_train, json_file)



