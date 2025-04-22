import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("🚀 Starting model load with LoRA using GPU + CPU fallback...")

# Paths
base_model = "meta-llama/Llama-3.2-1B"
adapter_path = "best-qlora-llama3"

try:
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded!")

    print("🧠 Loading base model with device_map='auto'...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",                  # ✅ this enables smart offloading
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✅ Base model loaded!")

    print("🔌 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, adapter_path)
    print("✅ LoRA adapter loaded!")

    # Send to same device as base model's input embedding layer
    device = next(model.base_model.parameters()).device

    # Test prompt
    prompt = "user: I feel anxious at night\nassistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )

    print("\n🧠 Assistant Response:\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    import traceback
    traceback.print_exc()
    input("\n❌ Press Enter to exit...")
