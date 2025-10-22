import glob
import os
import sys
from os.path import dirname

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, AutoTokenizer, Trainer
from transformers.trainer_pt_utils import save_state

import protein_llm
from protein_llm import PLLMConfig, PLLM
from protein_llm.data.data_collator import ProteinLLMChainDataCollator
from protein_llm.data.dataset import ProteinLLMChainDataset


def extract_key_params() -> str:
    """Extract all parameters from Hydra's override_dirname"""
    try:
        from hydra.core.hydra_config import HydraConfig
        hconf = HydraConfig.get()
        override_list = hconf.overrides.task
    except:
        return ""

    if len(override_list) == 0:
        return ""

    parts = []
    for item in override_list:
        parts.append(item.replace('=', '-'))
    return '_'.join(parts)


def get_unique_job_id() -> str:
    job_id = ""
    # slurm case
    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        job_id = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    # sge case
    if "JOB_ID" in os.environ:
        job_id = os.environ["JOB_ID"]
    if "SGE_TASK_ID" in os.environ and os.environ["SGE_TASK_ID"] != "undefined":
        job_id = f"{os.environ['JOB_ID']}_{os.environ['SGE_TASK_ID']}"
    # PJM case
    if "PJM_JOBID" in os.environ:
        job_id = os.environ["PJM_JOBID"]
    if "PBS_JOBID" in os.environ:
        job_id = os.environ["PBS_JOBID"]
    return job_id


@hydra.main(config_path=f"{dirname(protein_llm.__file__)}/../../config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    cfg: Contains all user configurations (base.yaml + arch/*.yaml + command line overrides)
         Does not include hydra.* configurations
    """
    from hydra.core.hydra_config import HydraConfig
    hconf = HydraConfig.get()

    # ============ Distributed training info ============
    rank = int(os.environ.get("RANK", 0))  # Global rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    job_id = get_unique_job_id()
    override_params_str = extract_key_params()

    output_name = f"{hconf.runtime.choices.model}"
    if job_id:
        output_name = f"{job_id}_{output_name}"
    if override_params_str:
        output_name += f"_{override_params_str}"
    output_dir = f"{dirname(protein_llm.__file__)}/../../results/{output_name}"

    # ============ 1. Hydra runtime info ============
    if rank == 0:  # Only let global rank 0 print
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nTrainingArguments output_dir: {output_dir}")

        print("=" * 60)
        print("Hydra Runtime Info")
        print("=" * 60)
        print(f"Job name: {hconf.job.name}")
        print(f"Output dir: {output_dir}")
        print(f"Working dir: {hconf.runtime.cwd}")
        print(f"Distributed: {is_distributed} (World size: {world_size}, Rank: {rank}, Local rank: {local_rank})")
        print(f"Selected model: {hconf.runtime.choices.model}")
        print(f"Command line overrides: {hconf.overrides.task}")

        print("\n" + "=" * 60)
        print("User Configuration (Merged base + arch)")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))

        # ============ 3. Save complete config (recommended for experiment logging) ============
        config_save_path = f"{output_dir}/config.yaml"
        with open(config_save_path, "w") as f:
            OmegaConf.save(cfg, f)
        print(f"\nConfig saved to: {config_save_path}")

    # ============ 4. Pass output_dir to TrainingArguments ============
    # TrainingArguments output_dir uses synchronized output directory
    training_args = TrainingArguments(
        output_dir=output_dir,
        **cfg.trainer
    )

    model_name = cfg.model.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dataset = ProteinLLMChainDataset(
        data_path=f"{cfg.dataset_dir}/{cfg.dataset}",
        tokenizer=tokenizer,
        # TODO: train_type=cfg.train_type
    )
    data_collator = ProteinLLMChainDataCollator(tokenizer=tokenizer)

    seq_token_id = tokenizer("<aa_seq>", add_special_tokens=False).input_ids
    assert len(seq_token_id) == 1
    seq_token_id = seq_token_id[-1]

    struct_token_id = tokenizer("<3d_struct>", add_special_tokens=False).input_ids
    assert len(struct_token_id) == 1
    struct_token_id = struct_token_id[-1]

    pllm_config = PLLMConfig(
        **cfg.model,
        seq_token_id=seq_token_id,
        struct_token_id=struct_token_id,
    )
    pllm = PLLM(pllm_config)
    pllm.load_protrek_weights()
    pllm.llm.resize_token_embeddings(len(tokenizer))  # IMPORTANT after adding tokens
    # TODO: pllm.freeze_params(cfg.freeze_choice)

    # ============ 5. Print model parameters info ============
    if rank == 0:
        print("\n" + "=" * 60)
        print("Model Parameters Summary")
        print("=" * 60)

        total_params = 0
        trainable_params = 0
        frozen_params = 0
        param_groups = {}

        for name, param in pllm.named_parameters():
            num_params = param.numel()
            total_params += num_params

            if param.requires_grad:
                trainable_params += num_params
                # Group statistics by module
                module_name = name.split('.')[0]
                if module_name not in param_groups:
                    param_groups[module_name] = {'trainable': 0, 'frozen': 0}
                param_groups[module_name]['trainable'] += num_params
            else:
                frozen_params += num_params
                module_name = name.split('.')[0]
                if module_name not in param_groups:
                    param_groups[module_name] = {'trainable': 0, 'frozen': 0}
                param_groups[module_name]['frozen'] += num_params

        def format_params(num):
            """Format parameter count"""
            if num >= 1e9:
                return f"{num/1e9:.2f}B"
            elif num >= 1e6:
                return f"{num/1e6:.2f}M"
            elif num >= 1e3:
                return f"{num/1e3:.2f}K"
            else:
                return str(num)

        print(f"\nTotal Parameters:     {format_params(total_params):>10} ({total_params:,})")
        print(f"Trainable Parameters: {format_params(trainable_params):>10} ({trainable_params:,})")
        print(f"Frozen Parameters:    {format_params(frozen_params):>10} ({frozen_params:,})")
        print(f"Trainable Ratio:      {trainable_params/total_params*100:>9.2f}%")

        print("\n" + "-" * 60)
        print("Parameters by Module:")
        print("-" * 60)
        print(f"{'Module':<20} {'Trainable':<15} {'Frozen':<15} {'Total':<15}")
        print("-" * 60)

        for module_name in sorted(param_groups.keys()):
            train_p = param_groups[module_name]['trainable']
            frozen_p = param_groups[module_name]['frozen']
            total_p = train_p + frozen_p
            print(f"{module_name:<20} {format_params(train_p):<15} {format_params(frozen_p):<15} {format_params(total_p):<15}")

        print("=" * 60 + "\n")

        print("Training Arguments:\n", training_args)

    trainer = Trainer(
        train_dataset=dataset,
        data_collator=data_collator,
        model=pllm,
        args=training_args,
    )

    ckpt_list = sorted(glob.glob(f"{training_args.output_dir}/checkpoint-*"))
    trainer.train(resume_from_checkpoint=len(ckpt_list) >= 1)
    save_state(trainer)


if __name__ == '__main__':
    # Filter out DeepSpeed launcher arguments that Hydra doesn't recognize
    sys.argv = [arg for arg in sys.argv if not arg.startswith('--local_rank')]
    main()
