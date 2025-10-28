import argparse
import asyncio
import copy
import json
import logging
import os
import random
from os.path import dirname, join, abspath, relpath, splitext
from typing import Dict

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage, ChatCompletions
from azure.core.credentials import AzureKeyCredential
from dotenv import find_dotenv, load_dotenv
from tqdm.asyncio import tqdm_asyncio, tqdm

import protein_eval
from protein_eval.utils.data import load_data, get_messages
from protein_eval.utils.parse import extract_last_answer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


DATA_ROOT = join(dirname(protein_eval.__file__), "../..", 'data')
RESULT_ROOT = join(dirname(protein_eval.__file__), "../..", 'eval_results')
CURRENT_DIR = dirname(abspath(__file__))


user_prompt_template = """
You are the Protein Insight Extractor, responsible for extracting key mechanistic and structural insights related only to the protein(s) named in the structure title.
Your role is to summarize the essential mechanistic, structural, and physicochemical findings from the paper in a concise, protein-focused form.

You are provided with:
- Structure Title: {structure_title} — specifies the target protein or complex of interest.
- Main Text: {main_text} — detailed scientific paper about protein structure, dynamics, and mechanisms.

Objective
Produce a compact, high-fidelity summary of the mechanistic content of the paper, grounded in the protein specified by the structure title.
This output will serve as the structured input for the Mechanistic QA Generator.

Instructions
1. Determine the Target Protein
    - Identify which protein (or subunit) in the {structure_title} is the main focus.
    - If the title includes a ligand or inhibitor (e.g., “enzyme X in complex with inhibitor Y”), concentrate on the protein (enzyme X), while incorporating relevant details about its interactions or bound ligands only as they relate to the protein’s structure and function.
2.	Extract Core Insights (≤ 10)
    From the {main_text}, identify up to 10 insights that describe:
    - How sequence features, motifs, or domains determine structure or activity.
    - How specific residues or physicochemical properties (charge, polarity, hydrophobicity, etc.) influence function.
    - How structural transitions (folding, conformational change, dimerization, etc.) relate to activity or regulation.
    - How chemical interactions (hydrogen bonds, salt bridges, π–π stacking, etc.) stabilize or enable function.
    - How binding, catalysis, or allostery emerge from the protein’s molecular architecture.
    - Any thermodynamic or kinetic findings (e.g., affinity, enthalpy, entropy, cooperativity).
    Constraints:
    - Insights should be mechanistic and concise, not general descriptions.
3. Keyword Identification
    - For each insight, extract a few (1–3) keywords that capture its molecular essence.

Output Format
Protein: [name derived from structure_title]

Insights:
1. [Concise mechanistic insight 1]
   Keywords: [k1, k2, k3]
2. [Concise mechanistic insight 2]
   Keywords: [k1, k2]
3. ...

Note:
- If the paper contains fewer than 10 meaningful insights, output only those with clear mechanistic and structural content.
- Avoid repeating the same theme (e.g., multiple insights about the same mutation unless mechanistically distinct).
""".strip()


async def run_model_by_message(
        client: ChatCompletionsClient, messages, data_id, model, gen_args: Dict,
        stream: bool = False, timeout: int = 300
):
    messages = copy.deepcopy(messages)
    for msg_idx, msg in enumerate(messages):
        msg_role = msg["role"]
        if msg_role == "system":
            messages[msg_idx] = SystemMessage(content=msg["content"])
        elif msg_role == "user":
            messages[msg_idx] = UserMessage(content=msg["content"])
        elif msg_role == "assistant":
            messages[msg_idx] = AssistantMessage(content=msg["content"])
        else:
            raise NotImplementedError(
                f"Unsupported role ({msg_role}) in your messages! Should be 'system', 'user', or 'assistant'."
            )

    n_sampling = gen_args.pop("n", 1)
    responses = []
    for _ in range(n_sampling):
        retry_num, break_flag = 0, False
        while not break_flag:
            try:
                run_response = await asyncio.wait_for(
                    client.complete(
                        model=model, messages=messages, stream=stream,
                        # model_extras={"stream_options": {"include_usage": True}},
                        **gen_args
                    ),
                    timeout=timeout
                )
                if stream:
                    resp_text = ""
                    async for update in tqdm(run_response, unit="tok", total=gen_args.get("max_tokens", 32768)):
                        if not (update.choices and update.choices[0].delta.content):
                            if update.choices and update.choices[0].finish_reason:
                                break
                            continue

                        try:
                            resp_text += update.choices[0].delta.content
                        except (AttributeError, TypeError) as e:
                            logger.error(f"Error {e} occurred for {model}. Return None for the current request.")
                            resp_text = None
                            break
                else:
                    try:
                        _ = run_response.choices[0].message.content
                        resp_text = run_response
                    except AttributeError:
                        logger.error(f"AttributeError occurred for {model}. Return None for the current request.")
                        resp_text = None

                if resp_text is not None:
                    responses.append(resp_text)
                    break_flag = True

            except (asyncio.TimeoutError, ConnectionError) as e:
                logger.error(f"Network exception occurred for {data_id}: {e}")

            except Exception as e:
                if (
                        "rate limit" in str(e).lower()
                        or "too many requests" in str(e).lower()
                        or "server error" in str(e).lower()
                        or "connection error" in str(e).lower()
                ):
                    logger.error(f"Sever error occurred for {data_id}: {e}")
                else:
                    logger.error(f"Other error occurred for {data_id}: {e}. skip it")
                    # break_flag = True

            if not break_flag:
                retry_num = retry_num + 1
                if retry_num > 10:
                    logger.info(f"Max Retry reached for {data_id}, skip it")
                    break_flag = True
                else:
                    wait_time = min(2 ** retry_num, 10)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

    return responses


async def process_item(client: ChatCompletionsClient, doc, args) -> Dict:
    messages = get_messages(doc)
    gen_args = dict(max_tokens=args.max_tokens, n=args.n_sampling)

    responses = await run_model_by_message(
        client=client, messages=messages, data_id=doc["id"], model=args.model_id, gen_args=gen_args, stream=args.stream
    )

    return dict(
        id=doc["id"],
        pdb_id=doc["pdb_id"],
        metadata=doc,
        request=json.dumps(messages, ensure_ascii=False),
        responses=responses,
    )


async def process_with_semaphore(client, data_item, args, semaphore):
    async with semaphore:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return await process_item(client, data_item, args)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="DeepSeek-R1")
    parser.add_argument('--data_path', type=str, default=join(DATA_ROOT, "validation_data_combined.json"))
    parser.add_argument('--n_sampling', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--debug_samples', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument("--n_thread", type=int, default=1)
    args = parser.parse_args()

    reformat_model_id = args.model_id.replace('/', '--')
    reformat_data_id = splitext(relpath(args.data_path, DATA_ROOT))[0].replace('/', '--')
    result_save_path = join(
        RESULT_ROOT,
        reformat_model_id,
        f"stream-{args.stream}_{args.n_sampling}_{args.max_tokens}",
        f"{reformat_data_id}.jsonl"
    )
    os.makedirs(dirname(result_save_path), exist_ok=True)

    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f.readlines():
                data_dict = json.loads(line.strip())
                if any(resp is None for resp in data_dict.get("responses")):
                    continue
                finish_ids.append(data_dict["id"])

    if len(finish_ids) > 0:
        print(f"[INFO] Resume from {len(finish_ids)} finished samples")
    else:
        print(f"[INFO] Start from scratch")

    env_ok = load_dotenv(find_dotenv(usecwd=True), override=False)
    if not env_ok:
        print("[WARN] No .env found; using existing environment only.")
    else:
        print("[INFO] Successfully load .env!")

    client = ChatCompletionsClient(
        credential=AzureKeyCredential(os.environ.get("AZURE_OPENAI_KEY")),
        endpoint=os.environ.get("AZURE_OPENAI_BASE"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
    )
    print(f"[INFO] AzureOpenAI ChatCompletionsClient created: {client}")

    ds = load_data(args.data_path)
    ds = [doc for doc in ds if doc["id"] not in finish_ids]

    if len(ds) > 0:
        try:
            tasks = []
            semaphore = asyncio.Semaphore(args.n_thread)
            for data_item in ds:
                tasks.append(
                    process_with_semaphore(
                        client=client,
                        data_item=data_item,
                        args=args,
                        semaphore=semaphore
                    )
                )

            with open(result_save_path, "w" if len(finish_ids) == 0 else "a") as f:
                for task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                    ret_dict = await task

                    dump_dict = {
                        "id": ret_dict["id"],
                        "pdb_id": ret_dict["pdb_id"],
                        "request": ret_dict["request"],
                        "metadata": ret_dict["metadata"],
                        "responses": [],
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                    }

                    if ret_dict["responses"] is not None:
                        for single_response in ret_dict["responses"]:
                            if isinstance(single_response, ChatCompletions):
                                generated_text = single_response.choices[0].message.content
                                single_input_tokens = single_response.usage.prompt_tokens
                                single_output_tokens = single_response.usage.completion_tokens
                            elif isinstance(single_response, str):
                                generated_text = single_response
                                single_input_tokens = 0
                                single_output_tokens = 0
                            else:
                                raise RuntimeError("")

                            dump_dict["responses"].append(generated_text)
                            dump_dict["total_input_tokens"] += single_input_tokens
                            dump_dict["total_output_tokens"] += single_output_tokens

                        if len(dump_dict["responses"]) < args.n_sampling:
                            remain_num = args.n_sampling - len(dump_dict["responses"])
                            dump_dict["responses"].extend([None] * remain_num)

                    else:
                        dump_dict["responses"] = [None] * args.n_sampling

                    f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
                    f.flush()

        finally:
            if client and hasattr(client, 'close'):
                await client.close()

            # 清理未完成任务
            for task in asyncio.all_tasks(asyncio.get_running_loop()):
                if task is not asyncio.current_task() and not task.done():
                    task.cancel()

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
        for resp_text in doc.get("responses"):
            if resp_text is None:
                resp_ans, resp_lvl = None, "error"
            else:
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
    asyncio.run(main())

