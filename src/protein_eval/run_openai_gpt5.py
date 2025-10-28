import argparse
import asyncio
import json
import logging
import os
import random
from os.path import dirname, abspath, join, basename, splitext, relpath
from typing import Dict

from openai import AsyncAzureOpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv, find_dotenv

import protein_eval
from protein_eval.utils.data import load_data, get_messages
from protein_eval.utils.parse import extract_last_answer


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


DATA_ROOT = join(dirname(protein_eval.__file__), "../..", 'data')
RESULT_ROOT = join(dirname(protein_eval.__file__), "../..", 'eval_results')
CURRENT_DIR = dirname(abspath(__file__))


async def run_model_by_message(client: AsyncAzureOpenAI | AsyncOpenAI, messages, data_id, model, gen_args: Dict, args):
    retry_num, break_flag = 0, False
    responses = []
    while not break_flag:
        try:
            loop_num = args.n_sampling - len(responses)
            for _ in range(loop_num):
                single_response = await asyncio.wait_for(
                    client.responses.create(
                        model=model,
                        input=messages,
                        reasoning={
                            "effort": args.reasoning_effort,
                        },
                        text={
                            "verbosity": args.text_verbosity,
                        },
                        **gen_args
                    ),
                    timeout=300
                )
                # Extract model's text output
                # output_text = single_response.output_text
                responses.append(single_response)

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
                # 速率限制错误，需要较长时间重试
                logger.error(f"Sever error occurred for {data_id}: {e}")
            else:
                # 其他未知错误，记录并跳过
                logger.error(f"Other error occurred for {data_id}: {e}. skip it")
                break_flag = True

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


async def process_item(client: AsyncAzureOpenAI | AsyncOpenAI, doc, args) -> Dict:
    messages = get_messages(doc)

    max_tokens = args.max_tokens
    if args.model_id in ["gpt-4o", "o1"]:
        max_tokens = min(args.max_tokens, 16384)
    if args.model_id in ["o3", "o1", "o4-mini"]:
        gen_args = dict(max_completion_tokens=max_tokens)
    elif args.model_id in ["gpt-5-mini", "gpt-5-nano", "gpt-5"]:
        gen_args = dict(max_output_tokens=max_tokens)
    else:
        gen_args = dict(max_tokens=max_tokens)

    response = await run_model_by_message(
        client=client, messages=messages, data_id=doc["id"], model=args.model_id, gen_args=gen_args,
        args=args
    )

    return dict(
        id=doc["id"],
        pdb_id=doc["pdb_id"],
        metadata=doc,
        request=json.dumps(messages, ensure_ascii=False),
        responses=response,
    )


async def process_with_semaphore(client, data_item, args, semaphore):
    async with semaphore:
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return await process_item(client, data_item, args)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="gpt-5-mini")
    parser.add_argument('--data_path', type=str, default=join(DATA_ROOT, "validation_data_combined.json"))
    parser.add_argument('--reasoning_effort', type=str, default="medium")
    parser.add_argument('--text_verbosity', type=str, default="medium")
    parser.add_argument('--use_azure', action='store_true')
    parser.add_argument('--n_sampling', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=16384)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--n_thread", type=int, default=1)
    args = parser.parse_args()

    reformat_model_id = args.model_id.replace('/', '--')
    reformat_data_id = splitext(relpath(args.data_path, DATA_ROOT))[0].replace('/', '--')
    result_save_path = join(
        RESULT_ROOT,
        reformat_model_id,
        f"reason-{args.reasoning_effort}_text-{args.text_verbosity}_{args.n_sampling}_{args.max_tokens}",
        f"{reformat_data_id}.jsonl"
    )
    os.makedirs(dirname(result_save_path), exist_ok=True)

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

    env_ok = load_dotenv(find_dotenv(usecwd=True), override=False)
    if not env_ok:
        print("[WARN] No .env found; using existing environment only.")
    else:
        print("[INFO] Successfully load .env!")

    if args.use_azure:
        client = AsyncAzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_BASE"),
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", None),
            api_key=os.environ.get("AZURE_OPENAI_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        )
        print(f"[INFO] AzureOpenAI client created: {client}")
    else:
        client = AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE"),
            api_key=os.environ.get("OPENAI_KEY")
        )
        print(f"[INFO] OpenAI client created: {client}")

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
                            generated_text = single_response.output_text
                            dump_dict["responses"].append(generated_text)

                            single_input_tokens = single_response.usage.input_tokens
                            single_output_tokens = single_response.usage.output_tokens
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
        for resp_text in doc.get("responses", doc.get("response")):
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
