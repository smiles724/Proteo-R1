# leave the space for tuning prompt templates
import string

MM_PROMPT_DICT = {
    "vanilla": "{question}\n\nInputs:".strip(),

    "open-ended": """
You are a professional protein biologist.
You will receive a multi-chain protein in which each chain is described by two aligned tracks: (1) its amino-acid sequence (primary structure) and (2) a residue-wise discrete 3D structural track capturing local tertiary context.

Use both the sequence evidence and the structural track in a chain-aware manner to answer the open-ended question below:
{question}

First, reason step by step inside a <think></think> block, explicitly tying sequence-derived cues to structural signals.
Then, provide your final response inside an <answer></answer> block.

Multi-chain Protein Input:
""".strip(),

    "yes–no": """
You are a professional protein biologist.
You will receive a multi-chain protein in which each chain is described by two aligned tracks: (1) its amino-acid sequence (primary structure) and (2) a residue-wise discrete 3D structural track capturing local tertiary context.

Use both the sequence evidence and the structural track in a chain-aware manner to answer the yes-no question below:
{question}

First, reason step by step inside a <think></think> block, explicitly tying sequence-derived cues to structural signals.
Then, provide your final answer ("Yes" or "No") inside an <answer></answer> block.

Multi-chain Protein Input:
    """.strip(),

    "multiple-choice": """
You are a professional protein biologist.
You will receive a multi-chain protein in which each chain is described by two aligned tracks: (1) its amino-acid sequence (primary structure) and (2) a residue-wise discrete 3D structural track capturing local tertiary context.

Use both the sequence evidence and the structural track in a chain-aware manner to answer the multiple-choice question below:
{question}

First, reason step by step inside a <think></think> block, explicitly tying sequence-derived cues to structural signals.
Then, provide your final answer (the option letter of the correct one) inside an <answer></answer> block.

Multi-chain Protein Input:
    """.strip(),

    "zero-shot": """
You are a professional protein biologist.
You will receive a multi-chain protein in which each chain is described by its amino-acid sequence (primary structure).

Use both the sequence evidence in a chain-aware manner to answer the multiple-choice question below:
{question}

Provide your final answer (the option letter of the correct one) inside an <answer></answer> block.

Multi-chain Protein Input:
    """.strip(),
}


# wwPDB-style order
CHAIN_ID_CANDIDATES = string.ascii_uppercase + string.digits + string.ascii_lowercase
