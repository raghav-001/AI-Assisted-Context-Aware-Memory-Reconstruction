import yaml
import os
from smolagents import CodeAgent
from final_answer import FinalAnswerTool
from retriever_tool import RetrieverTool
import requests
import chromadb
from chromadb.config import Settings


from chromadb import PersistentClient
chroma_client = PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection("patient_conversations")


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

# Initializing imported tools
final_answer = FinalAnswerTool()
retriever_tool = RetrieverTool(collection=collection, embedding_fn=embedding_fn)

class Model:
    def __init__(self, model_name="llama3:latest", base_url="http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.base_url = base_url

    def __call__(self, messages, stop_sequences=None, grammar=None):
        ollama_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            ollama_messages.append({"role": role, "content": content})

        response = requests.post(
            self.base_url,
            json={
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": False  
            }
        )

        try:
            content = response.json().get("message", {}).get("content", "")
            # print("[DEBUG] Raw model output:")
            # print(content)
            return type("LLMResponse", (), {"content": content})
        except Exception as e:
            print("[Error decoding JSON from Ollama]")
            print("Raw response:\n", response.text)
            raise e



model = Model(model_name="llama3:latest")

# Load prompts
with open("prompts3.yaml", encoding="utf-8") as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    tools=[retriever_tool, final_answer],
    model=model,
    max_steps=1,
    add_base_tools=True,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="Health-care Assistant",
    description="An AI assistant that acts as a mental health support assistant.",
    prompt_templates=prompt_templates
)

def maintain_conversation():
    while True:
        user_input = input("User: ")
        response = agent.run(user_input, reset=False)
        print("Health-care assistant:", response)

        # Log to ChromaDB
        text_to_embed = f"{user_input}."
        embedding = embedding_fn.get_text_embedding(text_to_embed)
        collection.add(
            documents=[text_to_embed],
            embeddings=[embedding],
            ids=[f"conv-{os.urandom(6).hex()}"]
        )


maintain_conversation()