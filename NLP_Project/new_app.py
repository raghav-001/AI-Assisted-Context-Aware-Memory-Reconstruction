import yaml
import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from smolagents import CodeAgent
from final_answer import FinalAnswerTool
from retriever_tool import RetrieverTool
from chromadb import PersistentClient

# Optional: Clean up warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ────────────────────────────── Chroma Setup ──────────────────────────────
chroma_client = PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection("patient_conversations")

# ───────────── Embedding Function (Still using Ollama) ─────────────
def get_ollama_embedding(text, model="mxbai-embed-large"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    if response.status_code == 200:
        return response.json().get("embedding")
    else:
        raise Exception(f"Embedding failed: {response.status_code} - {response.text}")

class EmbeddingWrapper:
    def __init__(self, model_name="mxbai-embed-large"):
        self.model_name = model_name

    def get_text_embedding(self, text):
        return get_ollama_embedding(text, self.model_name)

embedding_fn = EmbeddingWrapper()

# ───────────────────────────── Tools Setup ─────────────────────────────
final_answer = FinalAnswerTool()
retriever_tool = RetrieverTool(collection=collection, embedding_fn=embedding_fn)

# ───────────── Load Base Model + LoRA Adapter ─────────────
print("🚀 Loading model + adapter...")
base_model = "meta-llama/Llama-3.2-1B"
adapter_path = "best-qlora-llama3"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",                # GPU+CPU fallback
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(base, adapter_path)
device = next(model.base_model.parameters()).device
print("✅ Model ready.")

# ───────────────────────────── CodeAgent Model Wrapper ───────────────
# class Model:
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer

#     def __call__(self, messages, stop_sequences=None, grammar=None):
#         prompt = ""
#         for m in messages:
#             role = m.get("role", "user")
#             content = m.get("content", "")
#             if isinstance(content, list):
#                 content = "".join(part.get("text", "") for part in content)
#             prompt += f"{role}: {content}\n"
#         prompt += "assistant:"

#         inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=200,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.9,
#                 eos_token_id=self.tokenizer.eos_token_id
#             )

#         result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return type("LLMResponse", (), {"content": result.split("assistant:")[-1].strip()})

class Model:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, messages, stop_sequences=None, grammar=None):
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            prompt += f"{role}: {content}\n"
        prompt += "assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = result.split("assistant:")[-1].strip()
        wrapped = f"```py\nfinal_answer(\"\"\"{response_text}\"\"\")\n```<end_code>"
        return type("LLMResponse", (), {"content": wrapped})



# ───────────────────────────── Load Prompt Templates ───────────────
with open("prompts3.yaml", encoding="utf-8") as stream:
    prompt_templates = yaml.safe_load(stream)

# ───────────────────────────── Agent Initialization ───────────────
agent = CodeAgent(
    tools=[retriever_tool, final_answer],
    model=Model(model, tokenizer),
    max_steps=1,
    add_base_tools=True,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="Health-care Assistant",
    description="An AI assistant that acts as a mental health support assistant.",
    prompt_templates=prompt_templates
)

# ───────────────────────────── Interaction Loop ───────────────
def maintain_conversation():
    while True:
        user_input = input("User: ")
        response = agent.run(user_input, reset=False)
        print("Health-care assistant:", response)

        # Store conversation to ChromaDB
        text_to_embed = f"{user_input}."
        embedding = embedding_fn.get_text_embedding(text_to_embed)
        collection.add(
            documents=[text_to_embed],
            embeddings=[embedding],
            ids=[f"conv-{os.urandom(6).hex()}"]
        )

if __name__ == "__main__":
    maintain_conversation()
