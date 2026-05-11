"""
Hyperparameter saving utilities for torch.nn.Module.

This module provides a standalone implementation of PyTorch Lightning's
save_hyperparameters() functionality for use with regular nn.Module classes.

Usage:
    class MyModel(nn.Module):
        def __init__(self, hidden_size, num_layers, dropout=0.1):
            super().__init__()
            save_hyperparameters(self)  # Saves all args to self._hparams
            # or
            save_hyperparameters(self, ignore=['some_arg'])  # Exclude specific args

    # Access hyperparameters:
    model = MyModel(256, 4)
    print(model.hparams)  # AttributeDict with hidden_size=256, num_layers=4, dropout=0.1
"""

import copy
import inspect
import types
from collections.abc import MutableMapping, Sequence
from typing import Any, Optional, Union


class AttributeDict(dict):
    """Extended dictionary accessible with dot notation.

    Example:
        >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
        >>> ad.key1
        1
        >>> ad.key1 = 2
        >>> ad['key1']
        2
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def __repr__(self) -> str:
        if not self:
            return f"{type(self).__name__}()"
        items = [f'"{k}": {v!r}' for k, v in self.items()]
        return "\n".join(items)


def _parse_class_init_keys(cls: type) -> tuple[str, Optional[str], Optional[str]]:
    """Parse key words for standard self, *args and **kwargs.

    Args:
        cls: The class to inspect.

    Returns:
        Tuple of (self_name, args_name, kwargs_name).
        args_name and kwargs_name may be None if not present.

    Example:
        >>> class Model:
        ...     def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
        ...         pass
        >>> _parse_class_init_keys(Model)
        ('self', 'my_args', 'my_kwargs')
    """
    init_parameters = inspect.signature(cls.__init__).parameters
    init_params = list(init_parameters.values())

    # self is always first
    n_self = init_params[0].name

    def _get_first_if_any(
        params: list[inspect.Parameter],
        param_type: inspect._ParameterKind,
    ) -> Optional[str]:
        for p in params:
            if p.kind == param_type:
                return p.name
        return None

    n_args = _get_first_if_any(init_params, inspect.Parameter.VAR_POSITIONAL)
    n_kwargs = _get_first_if_any(init_params, inspect.Parameter.VAR_KEYWORD)

    return n_self, n_args, n_kwargs


def _get_init_args(frame: types.FrameType) -> tuple[Optional[Any], dict[str, Any]]:
    """Extract __init__ arguments from a frame.

    Args:
        frame: The frame to extract arguments from.

    Returns:
        Tuple of (self_instance, args_dict).
        self_instance may be None if not in __init__.
    """
    _, _, _, local_vars = inspect.getargvalues(frame)

    if "__class__" not in local_vars or frame.f_code.co_name != "__init__":
        return None, {}

    cls = local_vars["__class__"]
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = _parse_class_init_keys(cls)

    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")

    # Only collect variables that appear in the signature
    local_args = {k: local_vars[k] for k in init_parameters if k in local_vars}

    # Expand **kwargs if present
    if kwargs_var and kwargs_var in local_args:
        local_args.update(local_args.get(kwargs_var, {}))

    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    self_arg = local_vars.get(self_var, None)

    return self_arg, local_args


def _collect_init_args(
    frame: types.FrameType,
    path_args: list[dict[str, Any]],
    inside: bool = False,
) -> list[dict[str, Any]]:
    """Recursively collect arguments passed to constructors in inheritance tree.

    Args:
        frame: The current stack frame.
        path_args: List of dictionaries containing constructor args.
        inside: Track if we are inside inheritance path.

    Returns:
        List of dictionaries with constructor arguments at each level.
    """
    _, _, _, local_vars = inspect.getargvalues(frame)

    if not isinstance(frame.f_back, types.FrameType):
        return path_args

    local_self, local_args = _get_init_args(frame)

    if "__class__" in local_vars:
        path_args.append(local_args)
        return _collect_init_args(frame.f_back, path_args, inside=True)

    if not inside:
        return _collect_init_args(frame.f_back, path_args, inside=False)

    return path_args


def save_hyperparameters(
    obj: Any,
    *args: Any,
    ignore: Optional[Union[Sequence[str], str]] = None,
    frame: Optional[types.FrameType] = None,
) -> None:
    """Save __init__ arguments to obj._hparams attribute.

    This function mimics PyTorch Lightning's save_hyperparameters() functionality
    for use with regular nn.Module classes.

    Args:
        obj: The object instance (typically self in __init__).
        *args: Specific argument names to save. If empty, saves all arguments.
        ignore: Argument name(s) to exclude from saving.
        frame: Stack frame to use. If None, uses caller's frame.

    Example:
        class MyModel(nn.Module):
            def __init__(self, hidden_size, num_layers, dropout=0.1):
                super().__init__()
                save_hyperparameters(self)
                # Now self._hparams contains all args
                # Access via self.hparams property or self._hparams

        class MyModel2(nn.Module):
            def __init__(self, hidden_size, num_layers, model_config):
                super().__init__()
                save_hyperparameters(self, ignore=['model_config'])
                # model_config is excluded
    """
    # Handle empty container case
    if len(args) == 1 and not isinstance(args[0], str) and not args[0]:
        return

    # Get the caller's frame
    if not frame:
        current_frame = inspect.currentframe()
        if current_frame:
            frame = current_frame.f_back

    if not isinstance(frame, types.FrameType):
        raise AttributeError("There is no `frame` available while being required.")

    # Collect init arguments from the call stack
    init_args = {}
    for local_args in _collect_init_args(frame, []):
        init_args.update(local_args)

    # Process ignore parameter
    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    elif isinstance(ignore, (list, tuple)):
        ignore = [arg for arg in ignore if isinstance(arg, str)]

    ignore = list(set(ignore))
    init_args = {k: v for k, v in init_args.items() if k not in ignore}

    # Determine what to save
    if not args:
        # Take all arguments
        hp = init_args
        obj._hparams_name = "kwargs" if hp else None
    else:
        # Take only listed arguments
        isx_non_str = [i for i, arg in enumerate(args) if not isinstance(arg, str)]
        if len(isx_non_str) == 1:
            hp = args[isx_non_str[0]]
            cand_names = [k for k, v in init_args.items() if v == hp]
            obj._hparams_name = cand_names[0] if cand_names else None
        else:
            hp = {arg: init_args[arg] for arg in args if isinstance(arg, str) and arg in init_args}
            obj._hparams_name = "kwargs"

    # Set hparams
    _set_hparams(obj, hp)

    # Make a deep copy for initial hyperparameters
    obj._hparams_initial = copy.deepcopy(obj._hparams)


def _set_hparams(obj: Any, hp: Union[MutableMapping, dict]) -> None:
    """Set hyperparameters on the object.

    Args:
        obj: The object to set hyperparameters on.
        hp: The hyperparameters dictionary.
    """
    if isinstance(hp, dict):
        hp = AttributeDict(hp)

    if hasattr(obj, '_hparams') and isinstance(obj._hparams, dict) and isinstance(hp, dict):
        obj._hparams.update(hp)
    else:
        obj._hparams = hp


class HParamsMixin:
    """Mixin class providing hparams property and save_hyperparameters method.

    This mixin can be used with nn.Module to provide Lightning-like
    hyperparameter handling.

    Usage:
        class MyModel(HParamsMixin, nn.Module):
            def __init__(self, hidden_size, num_layers):
                super().__init__()
                self.save_hyperparameters()

    Or use the standalone function:
        class MyModel(nn.Module):
            def __init__(self, hidden_size, num_layers):
                super().__init__()
                save_hyperparameters(self)
    """

    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        """The collection of hyperparameters saved with save_hyperparameters.

        Returns:
            Mutable hyperparameters dictionary.
        """
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        """The initial collection of hyperparameters (read-only copy).

        Returns:
            Immutable initial hyperparameters.
        """
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        return copy.deepcopy(self._hparams_initial)

    def save_hyperparameters(
        self,
        *args: Any,
        ignore: Optional[Union[Sequence[str], str]] = None,
        frame: Optional[types.FrameType] = None,
    ) -> None:
        """Save arguments to hparams attribute.

        Args:
            *args: Specific argument names to save, or a single dict/Namespace.
                   If empty, saves all __init__ arguments.
            ignore: Argument name(s) to exclude from saving.
            frame: Stack frame to use. If None, uses caller's frame.

        Example:
            # Save all arguments
            self.save_hyperparameters()

            # Save specific arguments
            self.save_hyperparameters('arg1', 'arg3')

            # Exclude specific arguments
            self.save_hyperparameters(ignore='arg2')
            self.save_hyperparameters(ignore=['arg2', 'arg4'])
        """
        # Get the caller's frame (need to go back one more level)
        if not frame:
            current_frame = inspect.currentframe()
            if current_frame:
                frame = current_frame.f_back

        save_hyperparameters(self, *args, ignore=ignore, frame=frame)
