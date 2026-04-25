import os
from datasets import load_dataset

def prepare_dataset(num_samples=20):
    """
    Loads the covid_qa_deepset dataset, extracts 20 QA pairs,
    and saves them locally for reproducible experiments.
    :param num_samples: Number of samples to test
    :return:
    """
    dataset = load_dataset("deepset/covid_qa_deepset", split="train")

    small_dataset = dataset.select(range(num_samples)).to_pandas()

    small_dataset["true_answer"] = small_dataset["answers"].apply(
        lambda x: x["text"][0] if len(x["text"]) > 0 else "No answer found"
    )

    final_df = small_dataset[["context", "question", "true_answer"]]

    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/covid_qa.csv"
    final_df.to_csv(save_path, index=False)

    return final_df

if __name__ == "__main__":
    prepare_dataset()