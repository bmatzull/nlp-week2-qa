import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import prepare_dataset
from src.model import ExtractiveQAPipeline

def run_extractive_task():
    data_path = "data/processed/covid_qa.csv"
    if not os.path.exists(data_path):
        df = prepare_dataset(20)
    else:
        df = pd.read_csv(data_path)

    models_to_test = [
        "deepset/roberta-base-squad2",
        "distilbert-base-uncased-distilled-squad"
    ]

    results = []

    for model_name in models_to_test:
        qa_system = ExtractiveQAPipeline(model_name)

        for index, row in df.iterrows():
            print(f"Processing Model: {model_name} | Question {index+1}/20")

            predicted_answer = qa_system.predict(
                question=row['question'],
                context=row['context']
            )

            results.append({
                "Model": model_name,
                "Question": row['question'],
                "Context": row['context'][:100] + "...",
                "True Answer": row['true_answer'],
                "Predicted Answer": predicted_answer
            })

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/task_extractive_results.csv", index=False)

if __name__ == "__main__":
    run_extractive_task()