import asyncio

from transformers import AutoTokenizer

from rllm.agents.protein_agent import ProteinAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import protein_reward_fn
from rllm.utils import compute_pass_at_k

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 64

    # Full PLLM Model with Protein Encoders
    # Start server: python ProteinFM/model/vllm_infer/serve_pllm_full.py --model-path ./pllm --port 30000
    # This serves the complete PLLM model with protein and structure encoders
    model_name = "pllm"  # Model identifier for the full PLLM server

    tokenizer = AutoTokenizer.from_pretrained("/mnt/efs/erran/rllm_v02/ProteinFM/model/pllm/llm")

    reward_fn = protein_reward_fn

    env_args = {
        "reward_fn": reward_fn,
    }

    sampling_params = {"temperature": 0.6, "top_p": 0.95}

    engine = AgentExecutionEngine(
        agent_class=ProteinAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "model": model_name,
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=2048,
        max_prompt_length=1024,
        n_parallel_agents=n_parallel_agents,
    )

    # Load protein test dataset (default: thermostability)
    dataset_name = os.getenv("PROTEIN_DATASET_NAME", "thermostability")
    test_dataset = DatasetRegistry.load_dataset(f"protein_test_{dataset_name}", "test")

    if test_dataset is None:
        print(f"Dataset not found, preparing {dataset_name} dataset...")
        from prepare_protein_data import prepare_rlvr_data

        _, test_dataset = prepare_rlvr_data(dataset_name=dataset_name)

    tasks = test_dataset.repeat(n=16)  # repeat to evaluate pass@k

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
