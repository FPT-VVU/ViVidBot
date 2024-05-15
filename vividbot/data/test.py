# # import ijson
# # import json

# # path = "/home/duytran/Downloads/vast27m_annotations/annotations.json"

# # # save first 1 million items
# # result = open("result.json", "w")
# # with open(path) as f:
# #     for i, item in enumerate(ijson.items(f, "item")):
# #         if i > 2000000:
# #             break
# #         result.write(json.dumps(item) + "\n")

# import google.generativeai as genai
from datasets import load_dataset
import os
import sys

sys.path.append(os.getcwd())
# from googletrans import Translator

# translator = Translator()

# safety_settings = [
#     {
#         "category": "HARM_CATEGORY_DANGEROUS",
#         "threshold": "BLOCK_NONE",
#     },
#     {
#         "category": "HARM_CATEGORY_HARASSMENT",
#         "threshold": "BLOCK_NONE",
#     },
#     {
#         "category": "HARM_CATEGORY_HATE_SPEECH",
#         "threshold": "BLOCK_NONE",
#     },
#     {
#         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#         "threshold": "BLOCK_NONE",
#     },
#     {
#         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#         "threshold": "BLOCK_NONE",
#     },
# ]

# genai.configure(api_key="AIzaSyDexs4YZqNvy6ij_qryTMaz3DyybH0pFKw")
# model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)


data = load_dataset("json", data_files=["/home/duytran/Downloads/vividbot_data/Vast2M.json"])

new_data = data["train"].select(range(1000))

# # def map_fn(batch):
# #     result = []
# #     for item in batch["vast_cap"]:
# #         response = model.generate_content(f"translate \"{item}\" from English to Vietnamese with natural tone and without grammar incorrect, just return the result of translation")
# #         print(response.text)
# #         result.append(response.text)
# #     batch["vast_cap"] = result
# #     return batch

# def map_fn(batch):
#     result = []
#     for item in batch["vast_cap"]:
#         response = translator.translate(item, src='en', dest='vi')
#         # print(response.text)
#         result.append(response.text)
#     batch["vast_cap"] = result
#     return batch

# data.map(map_fn, batched=True, batch_size=1000, num_proc=16)

from processor.translator import GGTranslator

trans = GGTranslator()
# new_data.map(trans.process, fn_kwargs={"key_translate": ["vast_cap", "audio_cap"]}, batched=True, batch_size=100)
def map_func(batch):
    result_translate = trans.process(batch["vast_cap"], src="en", dest="vi")
    batch["vast_cap"] = result_translate
    return batch
data.map(lambda x: map_func(x), batched=True, batch_size=100, num_proc=16, cache_file_name = "dataset_temp", load_from_cache_file=True)
