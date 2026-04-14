import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer

def standardize_for_vllm(input_dir, output_dir):
    print(f"Standardizing model from {input_dir} to {output_dir}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load model and tokenizer
    # We use Auto classes to ensure compatibility
    model = AutoModelForCausalLM.from_pretrained(input_dir)
    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    
    # Save as Safetensors (vLLM prefers this)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    print("Standardization complete.")
    print(f"Files in {output_dir}: {os.listdir(output_dir)}")

if __name__ == "__main__":
    print("\n=== TinyModooAI Model Conversion Tool ===")
    print("1: Pretrained Model (outputs/pretrained)")
    print("2: SFT Instruction Model (outputs/sft)")
    choice = input("Select model to convert (1/2): ")

    if choice == "2":
        source = "../outputs/sft"
        target = "../outputs/vllm_chat"
        model_name = "TinyModooAI-Chat"
    else:
        source = "../outputs/pretrained"
        target = "../outputs/vllm_base"
        model_name = "TinyModooAI-Base"
    
    if os.path.exists(source):
        print(f"\nTargeting: {model_name}")
        standardize_for_vllm(source, target)
        print(f"\nSUCCESS: {model_name} is ready for vLLM at {target}")
    else:
        print(f"\nERROR: Source directory {source} not found.")
        print("Please ensure the training or SFT script has completed successfully.")
