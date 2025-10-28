import argparse
import json
import os

from dataclasses import asdict
from os.path import dirname, abspath, join, relpath, splitext
from typing import Dict, List

from tqdm import tqdm
from vllm import EngineArgs, LLM

import protein_eval
from protein_eval.utils import build_model_config, ModelConfig
from protein_eval.utils.data import get_messages, load_data
from protein_eval.utils.parse import extract_last_answer
from protein_eval.utils.vllm import get_sampling_params


DATA_ROOT = join(dirname(protein_eval.__file__), "../..", 'data')
RESULT_ROOT = join(dirname(protein_eval.__file__), "../..", 'eval_results')
CURRENT_DIR = dirname(abspath(__file__))


def create_batch_inputs(data: List[Dict], model_config: ModelConfig, args):
    for i in range(0, len(data), args.batch_size):
        batch_data = data[i:i + args.batch_size]
        batch_inputs = []
        batch_metadata = []

        for doc in batch_data:
            messages = get_messages(doc)
            if args.no_chat_template:
                prompt_text = " ".join([msg["content"] for msg in messages])
            else:
                prompt_text = model_config.get_prompt_from_question(messages, enable_thinking=args.enable_thinking)

            if isinstance(prompt_text, str):
                inputs = {"prompt": prompt_text}
            else:
                inputs = prompt_text
            batch_inputs.append(inputs)
            batch_metadata.append(doc)

        yield batch_inputs, batch_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=join(DATA_ROOT, "validation_data_combined.json"))
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--no_chat_template', action='store_true')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--n_sampling', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=16384)
    parser.add_argument('--max_model_len', type=int, default=32768)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--enforce_eager', action='store_true')
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    sampling_params = get_sampling_params(
        model_id=args.model_id,
        override_sampling_params=dict(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            n=args.n_sampling,
            seed=args.seed,
        )
    )
    print(f"[INFO] Sampling params: {sampling_params}")

    reformat_model_id = args.model_id.replace('/', '--').rstrip('--')
    if args.enable_thinking:
        reformat_model_id += "-think"

    reformat_data_id = splitext(relpath(args.data_path, DATA_ROOT))[0].replace('/', '--')
    result_save_path = join(
        RESULT_ROOT, reformat_model_id,
        f"{sampling_params.temperature}_{sampling_params.top_p}_{sampling_params.n}_"
        f"{sampling_params.top_k}_{sampling_params.max_tokens}_seed{args.seed}",
        f"{reformat_data_id}.jsonl"
    )
    os.makedirs(dirname(result_save_path), exist_ok=True)
    with open(join(dirname(result_save_path), "sampling_params.txt"), "w") as f:
        f.write(str(sampling_params))

    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f.readlines():
                data_dict = json.loads(line.strip())
                finish_ids.append(data_dict["id"])

    if len(finish_ids) > 0:
        print(f"[INFO] Resume from {len(finish_ids)} finished samples")
    else:
        print(f"[INFO] Start from scratch")

    ds = load_data(args.data_path)
    ds = [doc for doc in ds if doc["id"] not in finish_ids]

    if len(ds) > 0:
        model_config = build_model_config(
            model_id=args.model_id, max_tokens=args.max_tokens, max_model_len=args.max_model_len
        )
        engine_args = model_config.default_engine_args
        if args.tensor_parallel_size is not None:
            engine_args["tensor_parallel_size"] = args.tensor_parallel_size
        if args.enforce_eager:
            engine_args["enforce_eager"] = args.enforce_eager
        engine_args = asdict(EngineArgs(**engine_args))
        print(f"[INFO] vLLM Engine arguments: {engine_args}")
        vlm_model = LLM(**engine_args)

        with open(result_save_path, "w" if len(finish_ids) == 0 else "a") as f:
            progress_bar = tqdm(total=len(ds))
            for batch_inputs, batch_metadata in create_batch_inputs(ds, model_config, args):
                if "prompt" in batch_inputs[0]:
                    outputs = vlm_model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
                else:
                    outputs = vlm_model.chat(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
                outputs = sorted(outputs, key=lambda o: int(o.request_id))

                for idx, output in enumerate(outputs):
                    dump_dict = dict(
                        id=batch_metadata[idx]["id"],
                        pdb_id=batch_metadata[idx]["pdb_id"],
                        request=batch_inputs[idx]["prompt"] if "prompt" in batch_inputs[idx] else json.dumps(batch_inputs[idx]),
                        metadata=batch_metadata[idx],
                        responses=[],
                    )

                    for sub_output in output.outputs:
                        generated_text = sub_output.text
                        dump_dict["responses"].append(generated_text)

                    f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
                    f.flush()
                    progress_bar.update(1)

            progress_bar.close()

    if os.path.getsize(result_save_path) == 0:
        print("[WARN] No results generated, skipping post-processing")
        return

    with open(result_save_path) as f:
        finished_data = [json.loads(line.strip()) for line in f.readlines()]
        finished_data = {doc["id"]: doc for doc in finished_data}
        finished_data = list(finished_data.values())

    final_data = []
    for doc in finished_data:
        doc_id = doc["id"]
        pdb_id = doc["pdb_id"]

        parsed_ans = []
        parsed_levels = []
        for resp_text in doc["responses"]:
            resp_ans, resp_lvl = extract_last_answer(resp_text)
            parsed_ans.append(resp_ans)
            parsed_levels.append(resp_lvl)

        gt = doc["metadata"]["agent2_qa_list"][0]["Ans"]["answer"]

        final_data.append(
            dict(
                id=doc_id,
                pdb_id=pdb_id,
                parsed_ans=parsed_ans,
                parsed_levels=parsed_levels,
                gt=gt
            )
        )

    with open(result_save_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
