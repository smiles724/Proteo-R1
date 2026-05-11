"""
AlphaFoldLRScheduler - Learning rate scheduler based on AlphaFold3.

Ported from upstream structure-design reference: src/boltz/model/optim/scheduler.py

Implements a learning rate schedule with:
1. Linear warmup from base_lr to max_lr
2. Plateau at max_lr
3. Exponential decay after start_decay_after_n_steps
"""

import torch


class AlphaFoldLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Implements the learning rate schedule defined in AF3.

    A linear warmup is followed by a plateau at the maximum
    learning rate and then exponential decay. Note that the
    initial learning rate of the optimizer in question is
    ignored; use this class' base_lr parameter to specify
    the starting point of the warmup.

    Schedule:
        - step 0 to warmup_no_steps: linear warmup from base_lr to max_lr
        - warmup_no_steps to start_decay_after_n_steps: plateau at max_lr
        - after start_decay_after_n_steps: exponential decay every decay_every_n_steps
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
        base_lr: float = 0.0,
        max_lr: float = 1.8e-3,
        warmup_no_steps: int = 1000,
        start_decay_after_n_steps: int = 50000,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        """Initialize the learning rate scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer.
        last_epoch : int, optional
            The last epoch (actually step number), by default -1
        verbose : bool, optional
            Whether to print verbose output, by default False
        base_lr : float, optional
            The base learning rate (start of warmup), by default 0.0
        max_lr : float, optional
            The maximum learning rate (end of warmup / plateau), by default 1.8e-3
        warmup_no_steps : int, optional
            The number of warmup steps, by default 1000
        start_decay_after_n_steps : int, optional
            The number of steps after which to start decay, by default 50000
        decay_every_n_steps : int, optional
            The number of steps between decay applications, by default 50000
        decay_factor : float, optional
            The multiplicative decay factor, by default 0.95
        """
        # Validate parameters
        step_counts = {
            "warmup_no_steps": warmup_no_steps,
            "start_decay_after_n_steps": start_decay_after_n_steps,
        }

        for k, v in step_counts.items():
            if v < 0:
                msg = f"{k} must be nonnegative"
                raise ValueError(msg)

        if warmup_no_steps > start_decay_after_n_steps:
            msg = "warmup_no_steps must not exceed start_decay_after_n_steps"
            raise ValueError(msg)

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_no_steps = warmup_no_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.decay_every_n_steps = decay_every_n_steps
        self.decay_factor = decay_factor

        # Note: PyTorch >= 2.0 changed verbose parameter type
        # We manually set verbose attribute instead of passing to super().__init__
        super().__init__(optimizer, last_epoch=last_epoch)

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.__dict__.update(state_dict)

    def get_lr(self) -> list:
        """Compute learning rate for current step.

        Returns
        -------
        list
            Learning rate for each parameter group (all same value).

        Raises
        ------
        RuntimeError
            If called outside of scheduler.step() context.
        """
        if not self._get_lr_called_within_step:
            msg = (
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )
            raise RuntimeError(msg)

        step_no = self.last_epoch

        if step_no <= self.warmup_no_steps:
            # Linear warmup: base_lr + (step/warmup) * max_lr
            lr = self.base_lr + (step_no / self.warmup_no_steps) * self.max_lr
        elif step_no > self.start_decay_after_n_steps:
            # Exponential decay
            steps_since_decay = step_no - self.start_decay_after_n_steps
            exp = (steps_since_decay // self.decay_every_n_steps) + 1
            lr = self.max_lr * (self.decay_factor ** exp)
        else:
            # Plateau at max_lr
            lr = self.max_lr

        return [lr for _ in self.optimizer.param_groups]
