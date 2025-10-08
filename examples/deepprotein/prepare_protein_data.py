from datasets import load_dataset
from rllm.data.dataset import DatasetRegistry


def prepare_rlvr_data(dataset_name="thermostability"):
    # Load your dataset (JSON/JSONL file or HF Hub repo)
    dataset = load_dataset("fangwu97/ProteinData", name=dataset_name)

    # Define preprocessing
    def preprocess_fn(example, idx):
        return {"prompt": example.get("prompt", ""), "aa_seq": example["sequence"], "split": example["set"], "ground_truth": example["target"], }

    dataset = dataset.map(preprocess_fn, with_indices=True)
    train_dataset = dataset[dataset['split'] != 'test']
    test_dataset = dataset[dataset['split'] == 'test']

    # Register datasets
    train_dataset = DatasetRegistry.register_dataset(f"protein_train_{dataset_name}", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset(f"protein_test_{dataset_name}", test_dataset, "test")

    return train_dataset, test_dataset


def prepare_sft_data():
    # Load your dataset (JSON/JSONL file or HF Hub repo)
    dataset = load_dataset("json", data_files="proteins.jsonl")["train"]

    # Define preprocessing
    def preprocess_fn(example, idx):
        return {"prompt": example.get("prompt", ""), "aa_seq": example["aa_seq"], "stru_str": example["stru_str"], "ground_truth": example["description"],
                # description is the target
                "data_source": "protein", }

    dataset = dataset.map(preprocess_fn, with_indices=True)

    # Split into train/test (e.g., 95/5 split)
    split_dataset = dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("protein_train", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("protein_test", test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_rlvr_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())
