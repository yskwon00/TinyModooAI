import os
import torch
from transformers import (
    MistralConfig,
    MistralForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets

def train():
    # 1. Choose Training Mode
    print("\n=== TinyModooAI Training Menu ===")
    print("1: New Training (Zero-base)")
    print("2: Continuous Training (Load from ./tinymodoo_final)")
    choice = input("Select option (1/2): ")

    if choice == "2":
        if os.path.exists("./tinymodoo_final"):
            print("Loading existing model from ./tinymodoo_final...")
            model = MistralForCausalLM.from_pretrained("./tinymodoo_final")
        else:
            print("Error: ./tinymodoo_final not found. Starting new training instead.")
            config = MistralConfig.from_json_file("config.json")
            model = MistralForCausalLM(config)
    else:
        print("Initializing new model from config.json...")
        config = MistralConfig.from_json_file("config.json")
        model = MistralForCausalLM(config)
    
    # 2. Setup Tokenizer (Option B: Using a high-performance pretrained tokenizer)
    print("Loading tokenizer...")
    dataset_cache_dir = "../datasets" # Local cache for datasets and models
    tokenizer_path = "mistralai/Mistral-7B-v0.1" 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=dataset_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load Mixed Dataset (English + Korean)
    print(f"Loading datasets into {dataset_cache_dir}...")
    
    # English: Wikitext (Random 1% sampling)
    ds_en_full = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
    ds_en = ds_en_full.shuffle().select(range(int(len(ds_en_full) * 0.01)))
    
    # Korean: Official Wikipedia (Random 1% sampling)
    ds_ko_full = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", cache_dir=dataset_cache_dir)
    ds_ko = ds_ko_full.shuffle().select(range(int(len(ds_ko_full) * 0.01)))
    
    # Ensure both datasets have the same column name "text"
    # ds_en usually has "text", ds_ko usually has "text" or "document"
    # Let's check and rename if necessary
    if "text" not in ds_ko.column_names:
        # If ds_ko has 'contents' or 'text', adjust accordingly
        # seopbo/wikipedia-ko-20231101 usually has 'text'
        pass

    # Mix and Split Dataset
    raw_dataset = concatenate_datasets([ds_en, ds_ko]).shuffle(seed=42)
    split_dataset = raw_dataset.train_test_split(test_size=0.1) # 10% for validation
    
    print(f"Total dataset size: {len(raw_dataset)} (Train: {len(split_dataset['train'])}, Eval: {len(split_dataset['test'])})")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_train = split_dataset["train"].map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)
    tokenized_eval = split_dataset["test"].map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)
    
    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 5. Training Arguments (Configured for "Best Model" tracking)
    training_args = TrainingArguments(
        output_dir="./outputs",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        # Optimization for best model
        eval_strategy="steps",     # Updated from evaluation_strategy
        eval_steps=50,             # Evaluate every 50 steps
        save_strategy="steps",     # Match evaluation strategy
        save_steps=50,             # Save every 50 steps
        save_total_limit=3,        # Keep only the 3 latest/best checkpoints
        load_best_model_at_end=True, # LOAD the best model (lowest loss) when finished
        metric_for_best_model="loss",
        greater_is_better=False,
        # Other settings
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save Final Model (This will be the BEST model because of load_best_model_at_end)
    print("Saving the best model to ../outputs/pretrained...")
    trainer.save_model("../outputs/pretrained")
    tokenizer.save_pretrained("../outputs/pretrained")

if __name__ == "__main__":
    train()
