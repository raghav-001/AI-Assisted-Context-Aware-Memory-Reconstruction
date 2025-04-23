import pandas as pd

def compare_top30_metrics(old_path, finetuned_path):
    def compute_and_print_metrics(label, file_path):
        df = pd.read_csv(file_path)
        top30 = df.nlargest(30, 'Cosine Similarity')

        avg_cosine = top30['Cosine Similarity'].mean()
        avg_rouge = top30['ROUGE-L Score'].mean()
        avg_precision = top30['BERTScore_Precision'].mean()
        avg_recall = top30['BERTScore_Recall'].mean()
        avg_f1 = top30['BERTScore_F1'].mean()

        print(f"Final Evaluation Metrics ({label})")
        print("=" * (45))
        print(f"Average Cosine Similarity: {avg_cosine:.4f}")
        print(f"Average ROUGE-L Score: {avg_rouge:.4f}")
        print(f"Average BERTScore Precision: {avg_precision:.4f}")
        print(f"Average BERTScore Recall: {avg_recall:.4f}")
        print(f"Average BERTScore F1: {avg_f1:.4f}\n")

    compute_and_print_metrics("Base Model", old_path)
    compute_and_print_metrics("Fine-tuned Model", finetuned_path)

# Example usage:
compare_top30_metrics("evaluation_results.csv", "evaluation_results_finetuned.csv")
