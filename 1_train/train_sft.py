import os
import torch
from transformers import (
    MistralForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

def train_sft():
    # 1. Load Pretrained Model and Tokenizer
    dataset_cache_dir = "../datasets" 
    model_path = "../outputs/pretrained" # Load from consolidated folder
    if not os.path.exists(model_path):
        print(f"Error: Pretrained model not found at {model_path}. Run train.py first.")
        return

    print(f"Loading base model from {model_path}...")
    model = MistralForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # SFT logic requires consistent padding

    # 2. Define Prompt Template (Alpaca style)
    prompt_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )

    # 3. Load Mixed Instruction Datasets (1% Random Sampling)
    print(f"Loading datasets into {dataset_cache_dir}...")
    
    # Common settings
    load_args = {"cache_dir": dataset_cache_dir}
    
    # (1) Alpaca - General
    ds_alpaca = load_dataset("yahma/alpaca-cleaned", split="train", **load_args).shuffle().select(range(500)) # Small fixed amount or % 
    
    # (2) Code - WizardLM/Code Style
    ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", **load_args).shuffle().select(range(200))
    
    # (3) Reasoning - GSM8K (Math)
    ds_math = load_dataset("gsm8k", "main", split="train", **load_args).shuffle().select(range(100))

    # Standardize and Combine
    print("Combining and standardizing datasets...")
    
    def map_alpaca(ex):
        query = f"{ex['instruction']}\n{ex['input']}".strip()
        return {"query": query, "output": ex["output"]}

    def map_code(ex):
        query = f"{ex['instruction']}\n{ex['input']}".strip()
        return {"query": query, "output": ex["output"]}

    def map_math(ex):
        return {"query": ex["question"], "output": ex["answer"]}

    processed_datasets = [
        ds_alpaca.map(map_alpaca, remove_columns=ds_alpaca.column_names),
        ds_code.map(map_code, remove_columns=ds_code.column_names),
        ds_math.map(map_math, remove_columns=ds_math.column_names),
    ]
    
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets(processed_datasets).shuffle(seed=42)
    print(f"Total SFT dataset size: {len(combined_dataset)}")

    def tokenize_function(examples):
        queries = examples["query"]
        outputs = examples["output"]
        
        full_prompts = []
        labels = []
        
        for query, out in zip(queries, outputs):
            prompt = prompt_template.format(instruction=query)
            
            # Full text for calculation
            full_text = prompt + out + tokenizer.eos_token
            
            # Tokenize
            tokenized = tokenizer(full_text, truncation=True, max_length=512)
            prompt_ids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
            
            # Create labels: mask original prompt with -100
            # Our goal is to train ONLY on the output (response)
            input_ids = tokenized["input_ids"]
            label = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
            
            # Ensure they are the same length
            label = label[:len(input_ids)] 
            
            full_prompts.append(input_ids)
            labels.append(label)
            
        return {"input_ids": full_prompts, "labels": labels}

    print("Preprocessing dataset...")
    tokenized_dataset = combined_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=combined_dataset.column_names
    ).train_test_split(test_size=0.1)

    # 4. Training Arguments (Optimized for Mac MPS Memory)
    training_args = TrainingArguments(
        output_dir="../outputs/sft_logs",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,   # Reduced from 4
        gradient_accumulation_steps=4,   # Effective batch size of 4
        gradient_checkpointing=True,     # Memory saving trick
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.05,
        fp16=False,                     # Use default or set to True if on M1/M2/M3 with MPS
        bf16=False,
        push_to_hub=False,
    )

    # 5. Data Collator (Handles dynamic padding)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    # 7. Start Fine-tuning
    print("Starting SFT (Instruction Tuning)...")
    trainer.train()

    # 8. Save Final SFT Model
    print("Saving SFT model to ../outputs/sft...")
    trainer.save_model("../outputs/sft")
    tokenizer.save_pretrained("../outputs/sft")
    print("Done! SFT model saved to ../outputs/sft")

if __name__ == "__main__":
    train_sft()
