import pandas as pd

# Load the CSV file
df = pd.read_csv("evaluation_results_finetuned.csv")

# Extract top 30 records based on Cosine Similarity
top30 = df.nlargest(30, 'Cosine Similarity')

# Save the top 30 records to a CSV file
top30.to_csv("top_30_cosine_similarity_records.csv", index=False)
