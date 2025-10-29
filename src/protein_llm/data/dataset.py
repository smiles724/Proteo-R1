import json
import string
from os.path import dirname

from typing import Dict, List, Optional

import torch
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import protein_llm
from protein_llm.data import MM_PROMPT_DICT, CHAIN_ID_CANDIDATES
from protein_llm.data.data_collator import ProteinLLMChainDataCollator
from protein_llm.utils.pt import create_pretraining_data_with_masking
from protein_llm.utils.sft import create_sft_labels_with_masking


def parse_chains(chains: Dict):
    aa_seq_list = []
    stru_str_list = []
    chain_prompt = ""

    for chain_id in CHAIN_ID_CANDIDATES:
        if chain_id not in chains:
            continue

        if "aa_seq" not in chains[chain_id] or "threeDi_seq" not in chains[chain_id]:
            raise RuntimeError("")

        aa_seq_list.append(chains[chain_id]["aa_seq"])
        stru_str_list.append(chains[chain_id]["threeDi_seq"].lower())
        # no : after the letter to prevent joint tokenization (e.g., A: may be encoded different from A and :)
        chain_prompt += f"\n<chain>Chain {chain_id}<seq><aa_seq></seq><struct><3d_struct></struct></chain>"

    return (aa_seq_list, stru_str_list), chain_prompt.strip()


def parse_sft_doc_by_keys(doc: Dict):
    # multimodal training data
    if "chains" in doc:
        if "response" in doc:
            question = doc["question"]
            assistant_response = doc["response"]
            doc_type = "open-ended"
            user_prompt = MM_PROMPT_DICT[doc_type].format(question=question)

        elif "agent2_qa_list" in doc:
            if len(doc["agent2_qa_list"]) != 1:
                raise NotImplementedError("")

            question = doc["agent2_qa_list"][0]["Q"]

            think = doc["agent2_qa_list"][0]["Ans"]["think"]
            answer = doc["agent2_qa_list"][0]["Ans"]["answer"]
            assistant_response = "<think>\n" + think + "\n</think>\n\n<answer>\n" + answer + "\n</answer>"

            doc_type = doc["agent2_qa_list"][0]["type"]

            if doc_type == "yes–no":
                user_prompt = MM_PROMPT_DICT["multiple-choice"].format(question=question + "  A: Yes  B: No")

            elif doc_type == "multiple-choice":
                options = doc["agent2_qa_list"][0]["options"]
                options_str = ""
                for opt_idx in range(len(options)):
                    opt_letter = string.ascii_uppercase[opt_idx]
                    opt = options[opt_idx]
                    options_str += f"{opt_letter}: {opt}  " # double blanks after each option
                options_str = "  " + options_str.strip() # double blanks before the first option
                user_prompt = MM_PROMPT_DICT["multiple-choice"].format(question=question + options_str)

            else:
                raise ValueError("")

        else:
            raise RuntimeError("")

        chains = doc["chains"]
        if len(chains) == 0:
            raise RuntimeError(f"Empty chains for {doc['pdb_id']}")

        protein_info, chain_prompt = parse_chains(chains)
        user_prompt += "\n" + chain_prompt

    # text-only training data
    else:
        raise NotImplementedError("")

    return user_prompt, assistant_response, protein_info


def parse_pt_doc_by_keys(doc: Dict):
    # multimodal training data
    if "chains" in doc:
        if "response" in doc:
            assistant_response = doc["response"]

        elif "agent2_qa_list" in doc:
            if len(doc["agent2_qa_list"]) != 1:
                raise NotImplementedError("Multiple QA pairs have not been implemented yet~")
            assistant_response = doc["agent2_qa_list"][0]["Ans"]["answer"]

        else:
            raise RuntimeError("")

        chains = doc["chains"]
        if len(chains) == 0:
            raise RuntimeError(f"Empty chains for {doc['pdb_id']}")

        protein_info, user_prompt = parse_chains(chains)

    # text-only training data
    else:
        raise NotImplementedError("")

    return user_prompt, assistant_response, protein_info


class ProteinLLMChainDataset(Dataset):
    """

    mm data example:
    {
        "pdb_id": ,
        "question": ,
        "response": ,
        "chains": {
            "A": {
                "aa_seq": ,
                "threeDi_seq": ,
            },
            "B": ...
        }
    }

    """

    def __init__(
            self,
            data_path: List[str] | str,
            tokenizer,
            train_type: str = "sft",
            inference_mode: bool = False,
            max_chain_num: Optional[int] = None,
    ):
        if isinstance(data_path, str):
            data_path = [data_path]

        self.data = []
        for dp in data_path:
            if dp.endswith(".jsonl"):
                with open(dp, 'r') as f:
                    self.data.extend([json.loads(line) for line in f])
            elif dp.endswith(".json"):
                with open(dp, 'r') as f:
                    self.data.extend(json.load(f))
            else:
                raise RuntimeError("Unsupported data format")

        self.tokenizer = tokenizer
        self.train_type = train_type
        if self.train_type not in ["sft", "pt"]:
            raise ValueError("Invalid train_type")
        self.inference_mode = inference_mode
        self.max_chain_num = max_chain_num

        added_token_num = 0
        added_token_num += self.tokenizer.add_tokens("<chain>")
        added_token_num += self.tokenizer.add_tokens("</chain>")

        added_token_num += self.tokenizer.add_tokens("<seq>")
        added_token_num += self.tokenizer.add_tokens("<aa_seq>")
        added_token_num += self.tokenizer.add_tokens("</seq>")

        added_token_num += self.tokenizer.add_tokens("<struct>")
        added_token_num += self.tokenizer.add_tokens("<3d_struct>")
        added_token_num += self.tokenizer.add_tokens("</struct>")

        added_token_num += self.tokenizer.add_tokens("<think>")
        added_token_num += self.tokenizer.add_tokens("</think>")
        added_token_num += self.tokenizer.add_tokens("<answer>")
        added_token_num += self.tokenizer.add_tokens("</answer>")

        if added_token_num > 0:
            print(f"[WARN] {added_token_num} tokens have been added into vocabulary of your tokenizer! "
                  f"Please don't forget to call `resize_token_embeddings` of your model to expand the token embeddings!")

        # parse template name from the tokenizer class
        tok_class_name = tokenizer.__class__.__name__
        if tok_class_name.startswith("Qwen2"):
            template_name = "qwen"
        elif tok_class_name.startswith("Qwen3"):
            template_name = "qwen3"
        else:
            raise NotImplementedError(tok_class_name)

        data_args = DataArguments()
        data_args.template = template_name
        self.template = get_template_and_fix_tokenizer(tokenizer, data_args)

        self.lengths = self.register_lengths()

    def register_lengths(self):
        """Calculate char-length for each data item to simulate its rough sequence length during training"""
        lengths = []
        for doc in tqdm(self.data, desc=f"Registering lengths of {self.__class__.__name__}"):
            if self.train_type == "sft":
                user_prompt, assistant_response, protein_info = parse_sft_doc_by_keys(doc)
            elif self.train_type == "pt":
                user_prompt, assistant_response, protein_info = parse_pt_doc_by_keys(doc)
            else:
                raise NotImplementedError("")

            aa_seq_list, stru_str_list = protein_info
            aa_seq_len = sum(len(item) for item in aa_seq_list)
            stru_str_len = sum(len(item) for item in stru_str_list)

            lengths.append(aa_seq_len + stru_str_len + len(user_prompt) + len(assistant_response))

        return lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc = self.data[idx]
        if self.max_chain_num is not None:
            new_chains = {}
            for chain_id in CHAIN_ID_CANDIDATES:
                if chain_id in doc["chains"]:
                    new_chains[chain_id] = doc["chains"][chain_id]
                if len(new_chains) == self.max_chain_num:
                    break
            doc["chains"] = new_chains

        if self.train_type == "sft":
            user_prompt, assistant_response, protein_info = parse_sft_doc_by_keys(doc)
            aa_seq_list, stru_str_list = protein_info

            if not self.inference_mode:
                encoded_pairs = self.template.encode_multiturn(
                    tokenizer=self.tokenizer,
                    messages=[dict(role="user", content=user_prompt), dict(role="assistant", content=assistant_response)]
                )

                input_ids, labels = create_sft_labels_with_masking(
                    encoded_pairs=encoded_pairs,
                    tokenizer_eos_token_id=self.tokenizer.eos_token_id,
                )

                input_ids, labels = torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)

            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    [dict(role="user", content=user_prompt)], tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt_text, return_tensors="pt")
                input_ids = inputs["input_ids"][0]
                attention_mask = inputs["attention_mask"][0]
                labels = torch.full_like(input_ids, fill_value=-100)

        elif self.train_type == "pt":
            user_prompt, assistant_response, protein_info = parse_pt_doc_by_keys(doc)
            aa_seq_list, stru_str_list = protein_info

            if not self.inference_mode:
                input_ids, labels = create_pretraining_data_with_masking(
                    tokenizer=self.tokenizer, user_text=user_prompt, assistant_text=assistant_response, add_eos=False
                )

            else:
                inputs = self.tokenizer(user_prompt, add_special_tokens=False, return_tensors="pt")
                input_ids = inputs["input_ids"][0]
                labels = torch.full_like(input_ids, fill_value=-100)

            attention_mask = torch.ones_like(input_ids)

        else:
            raise RuntimeError("Unsupported train_type")

        if "pdb_id" in doc:
            pdb_id = doc["pdb_id"]
        elif "protein_name" in doc:
            pdb_id = doc["protein_name"]
        else:
            raise RuntimeError("Unsupported protein_name")

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            aa_seq=aa_seq_list,
            stru_str=stru_str_list,
            metadata=dict(
                pdb_id=pdb_id
            )
        )


def test():
    data_root = f"{dirname(protein_llm.__file__)}/../../data"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", use_fast=True)
    dataset = ProteinLLMChainDataset(
        # data_path=f"{data_root}/Protdt_combined_data/af_structured_dataset.json",
        data_path=[
            f"{data_root}/pdb_sft_850k_1021.jsonl",
            f"{data_root}/selected_mcqa10_15k_1028.json",
        ],
        tokenizer=tokenizer,
        train_type="sft",
    )
    data_collator = ProteinLLMChainDataCollator(tokenizer=tokenizer, return_metadata=True, return_protein_idx=True)

    dataloader = DataLoader(
        dataset, batch_size=64, collate_fn=data_collator, drop_last=False, shuffle=False,
        num_workers=16,
        # num_workers=0,
    )

    pdb_id2len = {}
    for batch in tqdm(dataloader):
        valid_lens = batch["attention_mask"].sum(dim=-1)
        valid_lens = valid_lens.tolist()
        pdb_ids = [doc["pdb_id"] for doc in batch["metadata"]]

        for idx, (pdb_id, valid_len) in enumerate(zip(pdb_ids, valid_lens)):
            protein_pos = torch.where(batch["protein_idx"] == idx)[0]

            protein_len = 0
            for p_pos in protein_pos:
                protein_len += len(batch["aa_seq"][p_pos])
                protein_len += len(batch["stru_str"][p_pos])

            pdb_id2len[pdb_id] = valid_len + protein_len

    with open(f"{data_root}/pdb_and_selected_id2len.json", "w") as f:
        json.dump(pdb_id2len, f, indent=4)

    # Statistical analysis
    import numpy as np
    import matplotlib.pyplot as plt

    lengths = list(pdb_id2len.values())
    lengths_array = np.array(lengths)

    print("\n" + "=" * 60)
    print("Sequence Length Statistics")
    print("=" * 60)
    print(f"Total samples: {len(lengths)}")
    print(f"Min length:    {np.min(lengths_array)}")
    print(f"Max length:    {np.max(lengths_array)}")
    print(f"Mean:          {np.mean(lengths_array):.2f}")
    print(f"Median:        {np.median(lengths_array):.2f}")
    print(f"25% quantile:  {np.percentile(lengths_array, 25):.2f}")
    print(f"75% quantile:  {np.percentile(lengths_array, 75):.2f}")
    print(f"90% quantile:  {np.percentile(lengths_array, 90):.2f}")
    print(f"95% quantile:  {np.percentile(lengths_array, 95):.2f}")
    print(f"99% quantile:  {np.percentile(lengths_array, 99):.2f}")
    print(f"Std deviation: {np.std(lengths_array):.2f}")
    print("=" * 60)

    # Histogram visualization
    plt.figure(figsize=(12, 6))

    # Plot 1: Full distribution
    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Sequence Length Distribution (Full)')
    plt.axvline(np.median(lengths_array), color='r', linestyle='--', label=f'Median: {np.median(lengths_array):.0f}')
    plt.axvline(np.mean(lengths_array), color='g', linestyle='--', label=f'Mean: {np.mean(lengths_array):.0f}')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 2: Zoomed distribution (up to 95th percentile)
    plt.subplot(1, 2, 2)
    p95 = np.percentile(lengths_array, 95)
    filtered_lengths = [l for l in lengths if l <= p95]
    plt.hist(filtered_lengths, bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title(f'Sequence Length Distribution (≤95th percentile: {p95:.0f})')
    plt.axvline(np.median(lengths_array), color='r', linestyle='--', label=f'Median: {np.median(lengths_array):.0f}')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{data_root}/sequence_length_distribution.png", dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved to: {data_root}/sequence_length_distribution.png")


if __name__ == '__main__':
    test()

