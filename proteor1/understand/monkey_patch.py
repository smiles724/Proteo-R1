from loguru import logger
from transformers import PreTrainedModel


def apply_liger_kernel_to_proteor1_understand(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel | None = None,
    use_rmpad: bool = False,
) -> None:
    """OSS builds do not ship the legacy upstream-trainer monkey-patch registry."""
    logger.warning("ProteoR1 understand Liger monkey patch is not registered in the OSS package.")


apply_liger_kernel_to_protenix_qwen = apply_liger_kernel_to_proteor1_understand
