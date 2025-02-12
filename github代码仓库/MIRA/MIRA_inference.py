from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import csv


model_path = r'' # Your model path
lora_path = r'' # Your LoRA path
output_path = r'' # Your output path
df = pd.read_csv(r'', encoding='utf-8-sig')
output_file_name = '' # Your output file name
modality = '' # Your modality

instruction_prompt = '' # Your instruction



# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(lora_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

print('Loading Finished!')
print('\n')



def clean_response(text):
    # 1. 统一换行符为\n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. 分行处理每一行
    lines = []
    for line in text.split('\n'):
        # 去除每行首尾的空白字符
        line = line.strip()
        # 只保留非空行
        if line:
            # 确保每行结尾有标准的句号或分号
            if not line.endswith('。') and not line.endswith('；'):
                line = line + '。'
            lines.append(line)
    
    # 3. 用标准换行符重新连接
    return '\n'.join(lines)





# 模型推理多个fold

output_path_folder = os.path.join(output_path, modality)

IDs = df['ID'].values
prompts = df['report_description'].values
gts = df['report_diagnosis'].values


# 输出结果
Response_result_init_csv = pd.DataFrame(columns = ['ID', 'finding', 'gt', 'impression'])
Response_result_init_csv.to_csv(os.path.join(output_path_folder, output_file_name), index=False, encoding='utf-8-sig')


# 批量推理
for i, prompt in enumerate(tqdm(prompts)):

    # prompt = prompt_add + prompt

    inputs = tokenizer.apply_chat_template([{"role": "user", "content": instruction_prompt},{"role": "user", "content": str(prompt)}],
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
        cleaned_response = clean_response(Response_result)

    pd.DataFrame(np.column_stack((IDs[i], prompt, gts[i], cleaned_response))).to_csv(
        os.path.join(output_path_folder, output_file_name), 
        index=False, 
        mode='a', 
        header=False, 
        encoding='utf-8-sig',
        lineterminator='\n', 
        quoting=csv.QUOTE_ALL,  
        quotechar='"', 
        escapechar='\\' 
    )









