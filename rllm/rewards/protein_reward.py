"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to
validate answers when necessary.
"""

from rllm.globals import OAI_RM_MODEL, THOUGHT_DELIMITER_END
from rllm.rewards.math_utils.utils import extract_answer, grade_answer_mathd, grade_answer_sympy, mathd_normalize_answer
from rllm.rewards.reward_types import RewardConfig, RewardOutput, RewardType
from rllm.system_prompts import ORM_PROMPT
from rllm.utils import call_gemini_llm, call_oai_rm_llm

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import numpy as np

def sequence_identity(gt: str, pred: str, k=2.0, alpha=0.8):
    """Compute identity-based reward for amino acid sequences."""
    # Global alignment (Needleman–Wunsch)
    alignments = pairwise2.align.globalds(gt, pred, matlist.blosum62,
                                          -11, -1)  # gap_open, gap_extend
    aln_gt, aln_pred, score, start, end = alignments[0]

    # Identity fraction
    matches = sum(a == b for a, b in zip(aln_gt, aln_pred) if a != "-" and b != "-")
    length = len(aln_gt)
    R_id = matches / length

    # Length penalty
    R_len = np.exp(-k * abs(len(pred) - len(gt)) / len(gt))

    # Final reward (arithmetic blend)
    R = alpha * R_id + (1 - alpha) * R_len
    return float(R)


def grade_answer_IF(given_answer: str, ground_truth: str) -> float:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)
    print(ground_truth_normalized_mathd, "========\n", given_answer_normalized_mathd)

    return sequence_identity(ground_truth_normalized_mathd, given_answer_normalized_mathd)


class RewardProteinFn:
    """
    Reward function for evaluating mathematical answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        # Extract information from task_info
        problem = task_info.get("problem", "")
        model_response = action

        # Handle None or empty response -- reward = 0.0
        if model_response is None or model_response == "":
            print("DEBUG: Empty or None response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            if self.config.apply_format_reward:  #  -- reward = 0.0
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)   -- reward = 0.0
        ground_truths = task_info.get("ground_truth", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, str | float | int):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:   #  -- reward = 0.0
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:

            if self.config.bio_task is "function":
                is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
                if is_correct:
                    # Apply tool call bonus if applicable and answer is correct
                    reward = self.config.correct_reward
                    if task_info.get("has_toolcall", False):
                        reward += self.config.toolcall_bonus
                    return RewardOutput(reward=reward, is_correct=True)

            elif self.config.bio_task is "inverse_folding":
                aar = grade_answer_IF(model_answer, ground_truth)
                assert  0.0 <= aar <= 1.0
                return RewardOutput(reward=aar, is_correct=aar == 1.0)

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_bio_orm:
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception:
                    print("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue

        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)


def rllm_reward_fn_math(data_source: str, llm_solution: str, ground_truth: str | list[str], extra_info=None, **kwargs):
    """Evaluates mathematical solutions against ground truth answers.

    This function creates a reward function to evaluate mathematical solutions by comparing
    them against provided ground truth answers. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: Either a single string or list of strings containing valid answers
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution is deemed correct, False otherwise

    Example:
        >>> rllm_reward_fn_math("function", "-2.83", "-1.56", False)   # function prediction
        True
    """
    if extra_info is None:
        extra_info = {}
    reward_config = RewardConfig()
    reward_fn = RewardProteinFn(reward_config)

    # Convert to new format
    task_info = {"problem": None, "problem_type": RewardType.PROTEIN, "data_source": data_source, "ground_truth": ground_truth, **extra_info}

    reward_response = reward_fn(task_info, llm_solution)
    return reward_response


if __name__ == "__main__":
    reward = RewardProteinFn(RewardConfig())
    task_info = {
        "data_source": "",
        "problem": "What is the thermostability of this protein?",
        "problem_type": RewardType.PROTEIN,
        "ground_truth": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"],
        "has_toolcall": True,
    }
    action = "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."

    output = reward(task_info, action)
    print(output)
