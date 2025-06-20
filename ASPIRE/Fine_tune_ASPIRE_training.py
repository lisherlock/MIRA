from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import TaskType, get_peft_model, LoraConfig
import os


def process_func(example):
    MAX_LENGTH = 384  
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n{instruction_prompt}<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }




if "__main__" == __name__:

    json_path = ''
    model_path = ''
    instruction_prompt = ''


    df = pd.read_json(json_path)
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32, 
        lora_dropout=0.1
    )


    model = get_peft_model(model, config)
    model.print_trainable_parameters()


    args = TrainingArguments(
        output_dir="",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=1,
        save_steps=200,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=False
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    model.save_pretrained("")


