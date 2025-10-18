import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    },
    "worker_process_setup_hook": "rllm.patches.verl_patch_hook.setup",
}


def get_ppo_ray_runtime_env():
    """
    Return the PPO Ray runtime environment.
    Avoid repeating env vars already set in the driver env.
    """
    env_vars = PPO_RAY_RUNTIME_ENV["env_vars"].copy()
    for key in list(env_vars.keys()):
        if os.environ.get(key) is not None:
            env_vars.pop(key, None)

    # Ensure workers can import the local rllm package without packaging large directories.
    # We do this by explicitly propagating PYTHONPATH to include the project root.
    try:
        import rllm  # type: ignore

        rllm_dir = os.path.dirname(os.path.abspath(rllm.__file__))
        project_root = os.path.dirname(rllm_dir)
    except Exception:
        # Fallback if rllm is not importable yet; infer from this file location.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))

    current_py_path = os.environ.get("PYTHONPATH", "")
    if project_root not in current_py_path.split(":"):
        env_vars["PYTHONPATH"] = f"{project_root}:{current_py_path}" if current_py_path else project_root
    else:
        # Even if already present on driver, make sure workers also get it explicitly.
        env_vars["PYTHONPATH"] = current_py_path

    # Exclude large/local artifacts to stay under Ray's 512MiB packaging limit
    excludes = [
        ".git/**",
        "model/pllm/**",
        "model/ProTrek_650M/**",
        "outputs/**",
        "wandb/**",
        "checkpoints/**",
        "flash_attn-*.whl",
        "flashinfer_python-*.whl",
    ]

    return {
        # Do NOT set working_dir to avoid packaging multi-GB artifacts.
        "env_vars": env_vars,
        "excludes": excludes,
        "worker_process_setup_hook": PPO_RAY_RUNTIME_ENV["worker_process_setup_hook"],
    }
