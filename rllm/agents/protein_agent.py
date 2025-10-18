import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class ProteinAgent(BaseAgent):
    """
    A protein agent that analyzes protein sequences and predicts properties,
    following the BaseAgent interface.
    """

    def __init__(self, accumulate_thinking=True):
        """
        Initialize the ProteinAgent.

        Args:
            accumulate_thinking: If True, keep thinking tags in conversation history.
                               If False, remove thinking from all but the last message.
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Process environment feedback and update internal state.

        Args:
            observation: The observation from the environment. Can be:
                        - dict with task info (initial observation from reset)
                        - empty dict {} (reward update after step)
                        - None (reward update)
            reward: The reward from the environment
            done: Whether the episode is done
            info: Additional info from the environment
        """

        # Reward update for existing step (None OR empty dict)
        if observation is None or (isinstance(observation, dict) and observation == {}):
            if self.trajectory.steps:
                cur_step = self.get_current_state()
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info
            return

        # This is a new observation, create a new step
        if isinstance(observation, dict):
            # Format protein task into a question
            formatted_observation = self._format_protein_task(observation)
        elif isinstance(observation, str):
            formatted_observation = observation
        else:
            raise ValueError(f"Invalid observation type: {type(observation)}")

        self.messages.append({"role": "user", "content": formatted_observation})

        new_step = Step(observation=formatted_observation)
        self._trajectory.steps.append(new_step)

    def _format_protein_task(self, task: dict) -> str:
        """
        Format a protein task dictionary into a prompt string.

        Args:
            task: Dictionary containing protein task info with keys like:
                 - 'prompt': Optional task-specific prompt
                 - 'aa_seq': Amino acid sequence
                 - 'stru_seq': Optional structure sequence (3Di tokens)
                 - 'ground_truth': Target value (not shown to agent)

        Returns:
            Formatted prompt string for the agent
        """
        # Start with the base prompt if provided
        prompt_parts = []

        if task.get("prompt"):
            prompt_parts.append(task["prompt"])
        else:
            # Default protein analysis prompt
            prompt_parts.append("Analyze the following protein sequence and predict its property value.")

        # Add amino acid sequence
        if "aa_seq" in task:
            prompt_parts.append(f"\nAmino acid sequence: {task['aa_seq']}")
        elif "sequence" in task:
            prompt_parts.append(f"\nAmino acid sequence: {task['sequence']}")

        # Add structure sequence if available
        if "stru_seq" in task and task["stru_seq"]:
            prompt_parts.append(f"\nStructure sequence (3Di): {task['stru_seq']}")

        # Add instruction for response format
        prompt_parts.append("\nProvide your prediction as a numerical value within \\boxed{}.")

        return "\n".join(prompt_parts)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.

        Args:
            response: The model's response string

        Returns:
            Action object containing the response
        """

        # Update the latest step
        self.messages.append({"role": "assistant", "content": response})

        cur_step = self.get_current_state()
        cur_step.chat_completions = self.chat_completions
        cur_step.model_response = response

        # Parse thinking and action if present
        if response.count("</think>") == 1:
            thought, sep, action = response.partition("</think>")
            thought = thought + sep
            action = Action(action.strip())
        else:
            thought = None
            action = Action(response.strip())

        cur_step.thought = thought
        cur_step.action = action

        return action

    def reset(self) -> None:
        """Reset agent state for new episode (wipes trajectory and messages)."""
        self._trajectory = Trajectory()
        self.messages = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        Return conversation history for model interaction.

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        # Remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
