import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the evaluation results
df = pd.read_csv("evaluation_results.csv")

# Plot all key metrics together
plt.figure(figsize=(12, 6))

plt.plot(df["Cosine Similarity"], label="Cosine Similarity", marker='o')
plt.plot(df["ROUGE-L Score"], label="ROUGE-L Score", marker='s')
plt.plot(df["BERTScore_F1"], label="BERTScore F1", marker='^')

plt.title("Model Evaluation Metrics Across Examples")
plt.xlabel("Example Index")
plt.ylabel("Score")
plt.ylim(0, 1.05)  # Ensure all metrics fit in view
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


