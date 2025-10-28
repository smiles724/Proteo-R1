import json
from typing import Dict

from protein_llm.data import MM_PROMPT_DICT


def get_messages(doc: Dict):
    if len(doc["agent2_qa_list"]) != 1:
        raise RuntimeError("")

    question = doc["agent2_qa_list"][0]["Q"]
    user_prompt = MM_PROMPT_DICT["zero-shot"].format(question=question)

    chains = doc["chains"]
    for ch_id in chains:
        chains[ch_id].pop("threeDi_seq", None)
        assert len(chains[ch_id]) == 1, chains[ch_id]
    user_prompt += "\n" + json.dumps(chains, indent=4)

    return  [{"role": "user", "content": user_prompt}]

def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)
    elif data_path.endswith(".jsonl"):
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
    else:
        raise RuntimeError("")

    ds = []
    for doc in data:
        if isinstance(doc["agent2_qa_list"], list):
            for qa_idx, qa_item in enumerate(doc["agent2_qa_list"]):
                ds.append(
                    dict(
                        id=doc["pdb_id"] + f"_{qa_idx}",
                        pdb_id=doc["pdb_id"],
                        chains=doc["chains"],
                        agent2_qa_list=[qa_item],
                    )
                )
        elif isinstance(doc["agent2_qa_list"], dict):
            ds.append(
                dict(
                    id=doc["pdb_id"],
                    pdb_id=doc["pdb_id"],
                    chains=doc["chains"],
                    agent2_qa_list=[doc["agent2_qa_list"]],
                )
            )
        else:
            raise RuntimeError("")
    return ds
