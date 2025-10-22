from transformers import TrainerCallback

try:
    import deepspeed
    _HAS_DS = True
except Exception:
    deepspeed = None
    _HAS_DS = False


class DSInspectCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def _as_bool(self, val, default=False):
        # Supports: bool / callable -> bool / other types
        try:
            if callable(val):
                return bool(val())
            return bool(val)
        except Exception:
            return default

    def on_train_begin(self, args, state, control, **kwargs):
        tr = self.trainer
        print("[DS] has trainer? ", tr is not None)
        if tr is None:
            return

        # 1) DeepSpeed enabled status (compatible with both attribute and method forms)
        is_ds_attr = getattr(tr, "is_deepspeed_enabled", None)
        is_ds = self._as_bool(is_ds_attr, default=False)
        print("[DS] is_deepspeed_enabled:", is_ds)

        # 2) Wrapped model type
        print("[DS] wrapped model type:", type(tr.model))

        # 3) Try to read ZeRO stage (different versions have slightly different field/method names, with multiple fallbacks)
        stage = None
        try:
            eng = tr.model
            # Some versions use eng.zero_optimization() -> obj, others use eng.zero_optimization -> obj
            zero = getattr(eng, "zero_optimization", None)
            if callable(zero):
                zero = zero()
            if isinstance(zero, dict):
                stage = zero.get("stage", None)
            else:
                stage = getattr(zero, "stage", None)
        except Exception as e:
            print("[DS] read ZeRO stage failed:", repr(e))
        print(f"[DS] ZeRO stage: {stage}")

        # 4) Trainer-side key information
        print("[Trainer] deepspeed cfg path:", args.deepspeed)
        print("[Trainer] bf16:", args.bf16, "fp16:", args.fp16)
        print("[Trainer] per_device_train_batch_size:", args.per_device_train_batch_size)
        print("[Trainer] gradient_accumulation_steps:", args.gradient_accumulation_steps)

        # 5) Further verification (optional, only compared when deepspeed is installed)
        if _HAS_DS:
            try:
                print("[DS] isinstance DeepSpeedEngine:", isinstance(tr.model, deepspeed.DeepSpeedEngine))
            except Exception as e:
                print("[DS] isinstance check failed:", repr(e))


class DSInspectStep0(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step != 0:
            return
        tr = self.trainer
        # Key: use model_wrapped, not model
        wrapped = getattr(tr, "model_wrapped", tr.model)
        print("[DS][step0] type(model_wrapped):", type(wrapped))
        print("[DS][step0] isinstance DeepSpeedEngine:", isinstance(wrapped, deepspeed.DeepSpeedEngine))

        stage = None
        try:
            zero = getattr(wrapped, "zero_optimization", None)
            if callable(zero):
                zero = zero()
            if isinstance(zero, dict):
                stage = zero.get("stage", None)
            else:
                stage = getattr(zero, "stage", None)
        except Exception as e:
            print("[DS] read ZeRO stage failed:", repr(e))
        print("[DS] ZeRO stage (wrapped):", stage)
