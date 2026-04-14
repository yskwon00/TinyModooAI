import os
import subprocess
import sys
import urllib.request

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode

def quantize_gguf():
    print("\n=== TinyModooAI REAL GGUF Quantization Tool ===")
    
    # 1. Path Setup
    source_model = "../outputs/sft"
    output_base = "../outputs/quantized/gguf"
    os.makedirs(output_base, exist_ok=True)

    if not os.path.exists(source_model):
        print(f"Error: SFT model not found at {source_model}.")
        return

    # 2. Select Quantization Type
    print("\nSelect Quantization Level:")
    print("1: 4-bit (Q4_K_M) - Best balance, recommended for Mac")
    print("2: 8-bit (Q8_0)   - High quality, larger file")
    print("3: 16-bit (F16)   - Original quality, no compression")
    
    choice = input("Select (1/2/3): ")
    q_map = {"1": "Q4_K_M", "2": "Q8_0", "3": "F16"}
    q_method = q_map.get(choice, "Q4_K_M")
    
    temp_f16_path = os.path.join(output_base, "temp_f16.gguf")
    output_path = os.path.join(output_base, f"tinymodoo-sft-{q_method.lower()}.gguf")

    # 3. Check for llama-quantize (from 'brew install llama.cpp')
    if subprocess.run("command -v llama-quantize", shell=True).returncode != 0:
        print("\n[!] Error: 'llama.cpp' is not installed.")
        print("Please run: brew install llama.cpp")
        return

    # 4. Download convert_hf_to_gguf.py from a VERIFIED stable tag (b3600)
    # This version is highly compatible with gguf==0.10.0 and has the correct filename.
    convert_script = "convert_hf_to_gguf.py"
    
    # Force re-download to ensure we get the b3600 version
    if os.path.exists(convert_script):
        os.remove(convert_script)

    print("\nDownloading verified stable conversion script (tag b3600) from llama.cpp...")
    url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/b3600/convert_hf_to_gguf.py"
    try:
        urllib.request.urlretrieve(url, convert_script)
    except Exception as e:
        print(f"Failed to download from b3600: {e}")
        print("Trying alternative tag (b3500)...")
        url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/b3500/convert_hf_to_gguf.py"
        urllib.request.urlretrieve(url, convert_script)

    # 5. Check 'gguf' package version (vllm requires 0.10.0)
    print("\nEnsuring 'gguf' package is installed...")
    try:
        import gguf
    except ImportError:
        run_command(f"{sys.executable} -m pip install gguf==0.10.0")

    # 6. Step 1: Convert HF to F16 GGUF
    print("\n[Step 1/2] Converting HF to F16 GGUF...")
    ret = run_command(f"{sys.executable} {convert_script} {source_model} --outfile {temp_f16_path}")
    if ret != 0:
        print("Error during conversion to F16.")
        return

    if q_method == "F16":
        os.rename(temp_f16_path, output_path)
        print(f"\nSUCCESS: F16 model saved to {output_path}")
        return

    # 6. Step 2: Quantize F16 GGUF to Target Bit
    print(f"\n[Step 2/2] Quantizing to {q_method}...")
    ret = run_command(f"llama-quantize {temp_f16_path} {output_path} {q_method}")
    
    # Cleanup temp file
    if os.path.exists(temp_f16_path):
        os.remove(temp_f16_path)

    if ret == 0:
        print(f"\n✨ SUCCESS: Model quantized to {output_path}")
    else:
        print("\n[!] Quantization failed.")

if __name__ == "__main__":
    quantize_gguf()
