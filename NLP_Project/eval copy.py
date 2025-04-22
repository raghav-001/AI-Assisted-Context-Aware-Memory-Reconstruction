import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import evaluate
from statistics import mean
from new_app import agent  # Your assistant agent

# Load dataset
df = pd.read_csv("mentalchat16k.csv")[['input', 'output']].dropna().reset_index(drop=True)

# Load evaluation metrics
# bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bert_score = evaluate.load("bertscore")

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Embedding function
def get_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1)

# Cosine similarity function
def compute_similarity(ref, pred):
    emb1 = get_embedding(ref)
    emb2 = get_embedding(pred)
    sim = cosine_similarity(emb1, emb2)[0][0]
    print(f"Cosine Similarity Score: {sim:.4f}")
    return sim

# Storage
results = []
bert_p_list = []
bert_r_list = []
bert_f1_list = []


N = 20  # Number of examples to evaluate
df = df.sample(n=N).reset_index(drop=True)
for i in tqdm(range(N)):
    input_text = df.loc[i, 'input']
    reference = df.loc[i, 'output']
    # print(f"\n[Input {i}] {input_text}")
    print(f"Reference Output: {reference}")

    try:
        prediction = agent.run(input_text, reset=False)
        # print(f"[Generated Output]: {prediction}")

        # Cosine Similarity
        cosine_score = compute_similarity(reference, prediction)

        # BLEU & ROUGE-L
        # bleu_score = bleu.compute(predictions=[prediction], references=[[reference]])['bleu']
        rouge_score = rouge.compute(predictions=[prediction], references=[reference])['rougeL']
        # print(f"BLEU Score: {bleu_score:.4f}")
        print(f"ROUGE-L Score: {rouge_score:.4f}")

        # BERTScore
        bert_result = bert_score.compute(predictions=[prediction], references=[reference], lang="en", model_type="bert-base-uncased")
        precision = bert_result["precision"][0]
        recall = bert_result["recall"][0]
        f1 = bert_result["f1"][0]
        print(f"BERTScore -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        if rouge_score > 0.01 :
            # Append metrics
            results.append({
                "Input": input_text,
                "Reference Output": reference,
                "Generated Output": prediction,
                "Cosine Similarity": cosine_score,
                # "BLEU Score": bleu_score,
                "ROUGE-L Score": rouge_score,
                "BERTScore_Precision": precision,
                "BERTScore_Recall": recall,
                "BERTScore_F1": f1
            })

            bert_p_list.append(precision)
            bert_r_list.append(recall)
            bert_f1_list.append(f1)

    except Exception as e:
        print(f"[ERROR at index {i}]: {e}")
        continue

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results_finetuned.csv", index=False)

# Final average scores
print("\nFinal Evaluation Metrics")
print("========================")
print(f"Average Cosine Similarity: {results_df['Cosine Similarity'].mean():.4f}")
# print(f"Average BLEU Score: {results_df['BLEU Score'].mean():.4f}")
print(f"Average ROUGE-L Score: {results_df['ROUGE-L Score'].mean():.4f}")
print(f"Average BERTScore Precision: {mean(bert_p_list):.4f}")
print(f"Average BERTScore Recall: {mean(bert_r_list):.4f}")
print(f"Average BERTScore F1: {mean(bert_f1_list):.4f}")