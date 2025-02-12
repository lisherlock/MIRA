from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from tqdm import tqdm
import pandas as pd
import os
import numpy as np


model_path = r''
lora_path = r""
output_path = r""
csv_input_folder = r''



def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.bfloat16)
    return tokenizer, model

tokenizer, model = get_model()

print('Loading Finished!')
print('\n')




file_name = os.listdir(os.path.join(csv_input_folder))[0]
df = pd.read_csv(os.path.join(csv_input_folder, file_name), encoding='utf-8-sig')

IDs = df['ID'].values
gts = df['gt'].values
findings = df['finding'].values
impressions = df['impression'].values

prompt_instruction = ''

Response_result_init_csv = pd.DataFrame(columns = ['ID', 'finding', 'gt', 'impression', 'result'])
Response_result_init_csv.to_csv(os.path.join(output_path, file_name), index=False, encoding='utf-8-sig')


# 批量推理
for i, prompt in enumerate(tqdm(gts)):

    prompt = 'Report Description: ' + str(findings[i]) + '。Reference Report Impression: ' + str(gts[i]) + '。Output Report Impression: ' + str(impressions[i])

    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt_instruction},{"role": "user", "content": prompt}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True
                                        ).to('cuda')

    gen_kwargs = {"max_length": 8192, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        Response_result = str(tokenizer.decode(outputs[0], skip_special_tokens=True))

    pd.DataFrame(np.column_stack((IDs[i], findings[i], gts[i], impressions[i], Response_result))).to_csv(os.path.join(output_path, file_name), index=False, mode='a', header=False, encoding='utf-8-sig')







