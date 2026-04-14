import torch
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from typing import List, Dict, Optional

# 1. Setup FastAPI
app = FastAPI(title="TinyModooAI OpenAI-Compatible Server")

# 2. Load Model (CPU for stability on Mac)
model_path = "../outputs/sft"
model_id = "tinymodoo-ai-chat" # ID shown in Open WebUI

print(f"Loading model on Mac from {model_path}...")
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32,
    device_map={"": device}
)

# --- OpenAI Standard Schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "antigravity"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Convert OpenAI messages to Alpaca Prompt Template
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    for msg in request.messages:
        if msg.role == "user":
            prompt += f"### Instruction:\n{msg.content}\n\n"
    prompt += "### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=request.temperature
        )
    
    # Extract only the newly generated tokens
    new_tokens = output_tokens[0][len(inputs["input_ids"][0]):]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Standard OpenAI Response Format
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(inputs["input_ids"][0]),
            "completion_tokens": len(new_tokens),
            "total_tokens": len(inputs["input_ids"][0]) + len(new_tokens)
        }
    }

if __name__ == "__main__":
    print(f"OpenAI-Compatible Server started at http://localhost:8002")
    print(f"Model ID: {model_id}")
    uvicorn.run(app, host="0.0.0.0", port=8002)
