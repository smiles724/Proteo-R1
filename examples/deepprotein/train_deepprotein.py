import hydra

from rllm.agents.protein_agent import ProteinAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import protein_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset(f"protein_train_{config.data.dataset_name}", "train")
    test_dataset = DatasetRegistry.load_dataset(f"protein_test_{config.data.dataset_name}", "test")

    env_args = {"reward_fn": protein_reward_fn}

    trainer = AgentTrainer(
        agent_class=ProteinAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
