import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize_awq():
    print("\n=== TinyModooAI AWQ Quantization Tool (4-bit) ===")
    
    # 1. Path Setup
    model_path = "../outputs/sft"
    quant_path = "../outputs/quantized/awq"
    os.makedirs(quant_path, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Error: SFT model not found at {model_path}.")
        return

    # 2. Quantization Configuration
    quant_config = {
        "zero_point": True, 
        "q_group_size": 128, 
        "w_bit": 4, 
        "version": "GEMM"
    }

    # 3. Load Model & Tokenizer
    print(f"Loading model for AWQ quantization from {model_path}...")
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 4. Quantize (Calibration)
    # Note: For real use, you should provide a calibration dataset.
    # Here we use a standard approach for the pipeline verification.
    print("Starting AWQ Quantization (this requires GPU)...")
    try:
        model.quantize(tokenizer, quant_config=quant_config)
    except Exception as e:
        print(f"Quantization failed: {e}")
        print("Note: AWQ quantization usually requires an NVIDIA GPU and 'autoawq' library.")
        return

    # 5. Save Quantized Model
    print(f"Saving AWQ model to {quant_path}...")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    print(f"\nSUCCESS: AWQ model saved to {quant_path}")

if __name__ == "__main__":
    quantize_awq()
