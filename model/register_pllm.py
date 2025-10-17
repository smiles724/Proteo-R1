"""
Register PLLM with Transformers' AutoModel system for FSDP training.

This module should be imported via the external_lib config option in training scripts.
It registers the PLLM wrapper as a custom model type that can be loaded via
AutoModelForCausalLM.from_pretrained().

NOTE: This module does NOT import the full PLLM class to avoid import dependencies.
Instead, it lazy-loads PLLM only when from_pretrained() is actually called.
"""
import json
import os
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
import sys
from pathlib import Path


class PLLMConfig(PretrainedConfig):
    """
    Minimal config class for PLLM to satisfy AutoConfig requirements.
    """
    model_type = "protein_llm_wrapper"
    is_composition = False
    
    def __init__(self, **kwargs):
        # Store all config as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Required attributes for transformers
        if not hasattr(self, 'hidden_size'):
            self.hidden_size = kwargs.get('llm_hidden_size', 896)  # Default from Qwen2.5-0.5B
        
        # Standard transformers config attributes
        if not hasattr(self, 'tie_word_embeddings'):
            self.tie_word_embeddings = False  # PLLM doesn't tie embeddings
        
        if not hasattr(self, 'is_encoder_decoder'):
            self.is_encoder_decoder = False  # PLLM is decoder-only
        
        if not hasattr(self, 'vocab_size'):
            self.vocab_size = kwargs.get('llm_vocab_size', 151936)  # Default from Qwen2.5
        
        # For multimodal models, sub_configs can contain separate configs for different modalities
        # PLLM doesn't use this pattern, so we set it to an empty dict
        if not hasattr(self, 'sub_configs'):
            self.sub_configs = {}
        
        # Attention head configuration (from the underlying LLM)
        # These are needed by FSDP and monkey patching code
        if not hasattr(self, 'num_attention_heads'):
            self.num_attention_heads = kwargs.get('llm_num_attention_heads', 14)  # Default from Qwen2.5-0.5B
        
        if not hasattr(self, 'num_key_value_heads'):
            self.num_key_value_heads = kwargs.get('llm_num_key_value_heads', 2)  # Default from Qwen2.5-0.5B
        
        self.model_type = "protein_llm_wrapper"
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load config from a directory.
        Handles return_unused_kwargs parameter like standard Transformers configs.
        """
        # Check if we should return unused kwargs
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Try to load the LLM config to get attention head info
        llm_config_path = os.path.join(pretrained_model_name_or_path, "llm", "config.json")
        if os.path.exists(llm_config_path):
            with open(llm_config_path, 'r') as f:
                llm_config = json.load(f)
                # Extract attention head config from LLM
                config_dict.setdefault('llm_num_attention_heads', llm_config.get('num_attention_heads'))
                config_dict.setdefault('llm_num_key_value_heads', llm_config.get('num_key_value_heads'))
                config_dict.setdefault('llm_hidden_size', llm_config.get('hidden_size'))
                config_dict.setdefault('llm_vocab_size', llm_config.get('vocab_size'))
        
        # Create config instance
        config = cls(**{**config_dict, **kwargs})
        
        # Return based on return_unused_kwargs parameter
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Instantiate a PLLMConfig from a Python dictionary.
        This is required by Transformers' AutoConfig.from_pretrained().
        Handles return_unused_kwargs parameter.
        """
        # Check if we should return unused kwargs
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        
        # Merge config_dict with kwargs (kwargs take precedence)
        merged_config = {**config_dict, **kwargs}
        config = cls(**merged_config)
        
        # Return based on return_unused_kwargs parameter
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def get_text_config(self, decoder=True):
        """
        Return self as the text config.
        This is required by GenerationConfig.from_model_config().
        For PLLM, the wrapper itself is the text config.
        """
        return self


class PLLMForCausalLM(nn.Module):
    """
    Wrapper around PLLM that satisfies AutoModelForCausalLM interface.
    This allows FSDP to load and train PLLM models.
    
    NOTE: This class lazy-loads the actual PLLM implementation to avoid
    import errors during registration.
    """
    config_class = PLLMConfig
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize from a PLLMConfig or directly with kwargs.
        """
        # Initialize as nn.Module first
        super().__init__()
        
        # Lazy import PLLM to avoid dependency issues during registration
        from ProteinFM.model.proteinLLM_pllm import PLLM as _PLLM
        
        if config is not None:
            # Extract PLLM-specific args from config
            model_name = getattr(config, 'model_name', None)
            protein_config = getattr(config, 'protein_config', None)
            structure_config = getattr(config, 'structure_config', None)
            protrek_ckpt = getattr(config, 'protrek_ckpt', None)
            prot_slot = getattr(config, 'prot_slot', 1)
            stru_slot = getattr(config, 'stru_slot', 3)
            single_token_prefix = getattr(config, 'single_token_prefix', False)
            prefix_len = getattr(config, 'prefix_len', 4)
            proj_hid = getattr(config, 'proj_hid', 1024)
            dropout = getattr(config, 'dropout', 0.1)
            train_encoders = getattr(config, 'train_encoders', False)  # Encoders frozen by default for training
            dtype_str = getattr(config, 'dtype_str', 'auto')
        else:
            # Use kwargs directly
            model_name = kwargs.get('model_name')
            protein_config = kwargs.get('protein_config')
            structure_config = kwargs.get('structure_config')
            protrek_ckpt = kwargs.get('protrek_ckpt', None)
            prot_slot = kwargs.get('prot_slot', 1)
            stru_slot = getattr(config, 'stru_slot', 3)
            single_token_prefix = kwargs.get('single_token_prefix', False)
            prefix_len = kwargs.get('prefix_len', 4)
            proj_hid = kwargs.get('proj_hid', 1024)
            dropout = kwargs.get('dropout', 0.1)
            train_encoders = kwargs.get('train_encoders', False)
            dtype_str = kwargs.get('dtype_str', 'auto')
        
        # Initialize the underlying PLLM model
        pllm_model = _PLLM(
            model_name=model_name,
            protein_config=protein_config,
            structure_config=structure_config,
            protrek_ckpt=protrek_ckpt,
            prot_slot=prot_slot,
            stru_slot=stru_slot,
            single_token_prefix=single_token_prefix,
            prefix_len=prefix_len,
            proj_hid=proj_hid,
            dropout=dropout,
            train_encoders=train_encoders,
            dtype_str=dtype_str,
        )
        
        # Copy all PLLM attributes to this wrapper
        for attr_name in ['llm', 'protein_encoder', 'structure_encoder', 'prefix_mlp', 
                          'tokenizer', 'hidden_size', 'prefix_len', 'protein_config', 
                          'structure_config', 'train_encoders', 'proj_hid']:
            if hasattr(pllm_model, attr_name):
                setattr(self, attr_name, getattr(pllm_model, attr_name))
        
        # Copy methods
        for method_name in ['encode_protein_batch', 'forward', 'generate', 'save_pretrained']:
            if hasattr(pllm_model, method_name):
                setattr(self, method_name, getattr(pllm_model, method_name))
        
        # Store config
        self.config = config if config is not None else PLLMConfig(**kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        """
        Load PLLM from a saved directory.
        This overrides PLLM's from_pretrained to work with AutoModel.
        """
        print(f"[PLLMForCausalLM] Starting from_pretrained with path: {pretrained_model_name_or_path}")
        
        # Lazy import PLLM to avoid dependency issues
        from ProteinFM.model.proteinLLM_pllm import PLLM as _PLLM
        print(f"[PLLMForCausalLM] Imported PLLM class successfully")
        
        # Load using PLLM's from_pretrained
        print(f"[PLLMForCausalLM] Loading PLLM model from {pretrained_model_name_or_path}")
        pllm_model = _PLLM.from_pretrained(pretrained_model_name_or_path)
        print(f"[PLLMForCausalLM] PLLM model loaded successfully")
        
        # Load config
        if config is None:
            print(f"[PLLMForCausalLM] Loading config from {pretrained_model_name_or_path}")
            config = PLLMConfig.from_pretrained(pretrained_model_name_or_path)
            print(f"[PLLMForCausalLM] Config loaded successfully")
        
        # Create wrapper instance without calling __init__
        print(f"[PLLMForCausalLM] Creating wrapper instance")
        wrapped_model = cls.__new__(cls)
        
        # Initialize nn.Module parent class
        print(f"[PLLMForCausalLM] Initializing nn.Module parent class")
        nn.Module.__init__(wrapped_model)
        
        # Copy all PLLM attributes
        print(f"[PLLMForCausalLM] Copying PLLM attributes")
        for attr_name in ['llm', 'protein_encoder', 'structure_encoder', 'prefix_mlp', 
                          'tokenizer', 'hidden_size', 'prefix_len', 'protein_config', 
                          'structure_config', 'train_encoders', 'proj_hid']:
            if hasattr(pllm_model, attr_name):
                attr_value = getattr(pllm_model, attr_name)
                setattr(wrapped_model, attr_name, attr_value)
                print(f"[PLLMForCausalLM] Copied attribute: {attr_name} (type: {type(attr_value)})")
        
        # Register submodules as children for proper nn.Module behavior
        print(f"[PLLMForCausalLM] Registering submodules as children")
        if hasattr(wrapped_model, 'llm'):
            wrapped_model._modules['llm'] = wrapped_model.llm
            print(f"[PLLMForCausalLM] Registered llm as child module")
        if hasattr(wrapped_model, 'protein_encoder'):
            wrapped_model._modules['protein_encoder'] = wrapped_model.protein_encoder
            print(f"[PLLMForCausalLM] Registered protein_encoder as child module")
        if hasattr(wrapped_model, 'structure_encoder'):
            wrapped_model._modules['structure_encoder'] = wrapped_model.structure_encoder
            print(f"[PLLMForCausalLM] Registered structure_encoder as child module")
        if hasattr(wrapped_model, 'prefix_mlp'):
            wrapped_model._modules['prefix_mlp'] = wrapped_model.prefix_mlp
            print(f"[PLLMForCausalLM] Registered prefix_mlp as child module")
        
        # Verify that children are properly registered
        print(f"[PLLMForCausalLM] Children modules: {list(wrapped_model.named_children())}")
        direct_params = list(wrapped_model.named_parameters(prefix='', recurse=False))
        print(f"[PLLMForCausalLM] Direct parameters count: {len(direct_params)}")
        for name, param in direct_params[:3]:  # Print first 3 parameter names and shapes
            print(f"[PLLMForCausalLM] Direct param: {name} (shape: {param.shape})")
        
        # Copy methods
        print(f"[PLLMForCausalLM] Copying PLLM methods")
        for method_name in ['encode_protein_batch', 'forward', 'generate', 'save_pretrained']:
            if hasattr(pllm_model, method_name):
                setattr(wrapped_model, method_name, getattr(pllm_model, method_name))
                print(f"[PLLMForCausalLM] Copied method: {method_name}")
        
        # Set config
        wrapped_model.config = config
        print(f"[PLLMForCausalLM] Set config successfully")
        
        print(f"[PLLMForCausalLM] from_pretrained completed successfully")
        return wrapped_model
    
    def to(self, *args, **kwargs):
        """Delegate to() to the underlying LLM and encoders"""
        if hasattr(self, 'llm'):
            self.llm.to(*args, **kwargs)
        if hasattr(self, 'protein_encoder'):
            self.protein_encoder.to(*args, **kwargs)
        if hasattr(self, 'structure_encoder'):
            self.structure_encoder.to(*args, **kwargs)
        if hasattr(self, 'prefix_mlp'):
            self.prefix_mlp.to(*args, **kwargs)
        return self
    
    def cuda(self, device=None):
        """Delegate cuda() to the underlying modules"""
        return self.to('cuda' if device is None else f'cuda:{device}')
    
    def cpu(self):
        """Delegate cpu() to the underlying modules"""
        return self.to('cpu')
    
    def parameters(self, recurse=True):
        """Return parameters from all submodules"""
        params = []
        if hasattr(self, 'llm'):
            params.extend(self.llm.parameters(recurse=recurse))
        if hasattr(self, 'protein_encoder'):
            params.extend(self.protein_encoder.parameters(recurse=recurse))
        if hasattr(self, 'structure_encoder'):
            params.extend(self.structure_encoder.parameters(recurse=recurse))
        if hasattr(self, 'prefix_mlp'):
            params.extend(self.prefix_mlp.parameters(recurse=recurse))
        return iter(params)
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Return named parameters from all submodules"""
        print(f"[PLLMForCausalLM] named_parameters called with prefix='{prefix}', recurse={recurse}")
        memo = set() if remove_duplicate else None
        param_count = 0
        
        if not recurse:
            # When recurse=False, we should only return direct parameters
            # But our wrapper doesn't have direct parameters, so we need to return parameters from children
            print(f"[PLLMForCausalLM] recurse=False, getting parameters from direct children")
            for name, module in self.named_children():
                print(f"[PLLMForCausalLM] Processing child module: {name} (type: {type(module)})")
                # Get parameters from this child module with recurse=True to get all its parameters
                for param_name, param in module.named_parameters(prefix=f'{prefix}{name}.', recurse=True):
                    if memo is None or param not in memo:
                        if memo is not None:
                            memo.add(param)
                        param_count += 1
                        if param_count <= 5:  # Print first 5 parameters for debugging
                            print(f"[PLLMForCausalLM] Parameter {param_count}: {param_name} (shape: {param.shape})")
                        yield param_name, param
        else:
            # When recurse=True, get all parameters from all submodules
            print(f"[PLLMForCausalLM] recurse=True, getting parameters from all submodules")
            for name, module in [('llm', getattr(self, 'llm', None)),
                                 ('protein_encoder', getattr(self, 'protein_encoder', None)),
                                 ('structure_encoder', getattr(self, 'structure_encoder', None)),
                                 ('prefix_mlp', getattr(self, 'prefix_mlp', None))]:
                if module is not None:
                    print(f"[PLLMForCausalLM] Processing module: {name} (type: {type(module)})")
                    for param_name, param in module.named_parameters(prefix=f'{prefix}{name}.', recurse=True):
                        if memo is None or param not in memo:
                            if memo is not None:
                                memo.add(param)
                            param_count += 1
                            if param_count <= 5:  # Print first 5 parameters for debugging
                                print(f"[PLLMForCausalLM] Parameter {param_count}: {param_name} (shape: {param.shape})")
                            yield param_name, param
        
        print(f"[PLLMForCausalLM] named_parameters completed, total params: {param_count}")
    
    def train(self, mode=True):
        """Set training mode"""
        if hasattr(self, 'llm'):
            self.llm.train(mode)
        if hasattr(self, 'protein_encoder'):
            self.protein_encoder.train(mode)
        if hasattr(self, 'structure_encoder'):
            self.structure_encoder.train(mode)
        if hasattr(self, 'prefix_mlp'):
            self.prefix_mlp.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        return self
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'gradient_checkpointing_disable'):
            self.llm.gradient_checkpointing_disable()
        return self
    
    def state_dict(self, *args, **kwargs):
        """Get state dict from all submodules"""
        state = {}
        if hasattr(self, 'llm'):
            state.update({f'llm.{k}': v for k, v in self.llm.state_dict(*args, **kwargs).items()})
        if hasattr(self, 'protein_encoder'):
            state.update({f'protein_encoder.{k}': v for k, v in self.protein_encoder.state_dict(*args, **kwargs).items()})
        if hasattr(self, 'structure_encoder'):
            state.update({f'structure_encoder.{k}': v for k, v in self.structure_encoder.state_dict(*args, **kwargs).items()})
        if hasattr(self, 'prefix_mlp'):
            state.update({f'prefix_mlp.{k}': v for k, v in self.prefix_mlp.state_dict(*args, **kwargs).items()})
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into all submodules"""
        # Separate state dict by module prefix
        llm_state = {k.replace('llm.', ''): v for k, v in state_dict.items() if k.startswith('llm.')}
        protein_state = {k.replace('protein_encoder.', ''): v for k, v in state_dict.items() if k.startswith('protein_encoder.')}
        structure_state = {k.replace('structure_encoder.', ''): v for k, v in state_dict.items() if k.startswith('structure_encoder.')}
        prefix_state = {k.replace('prefix_mlp.', ''): v for k, v in state_dict.items() if k.startswith('prefix_mlp.')}
        
        if hasattr(self, 'llm') and llm_state:
            self.llm.load_state_dict(llm_state, strict=strict)
        if hasattr(self, 'protein_encoder') and protein_state:
            self.protein_encoder.load_state_dict(protein_state, strict=strict)
        if hasattr(self, 'structure_encoder') and structure_state:
            self.structure_encoder.load_state_dict(structure_state, strict=strict)
        if hasattr(self, 'prefix_mlp') and prefix_state:
            self.prefix_mlp.load_state_dict(prefix_state, strict=strict)
        return self
    
    def modules(self):
        """Return all submodules"""
        mods = [self]
        if hasattr(self, 'llm'):
            mods.extend(self.llm.modules())
        if hasattr(self, 'protein_encoder'):
            mods.extend(self.protein_encoder.modules())
        if hasattr(self, 'structure_encoder'):
            mods.extend(self.structure_encoder.modules())
        if hasattr(self, 'prefix_mlp'):
            mods.extend(self.prefix_mlp.modules())
        return iter(mods)
    
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Return all named submodules"""
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in [('llm', getattr(self, 'llm', None)),
                                 ('protein_encoder', getattr(self, 'protein_encoder', None)),
                                 ('structure_encoder', getattr(self, 'structure_encoder', None)),
                                 ('prefix_mlp', getattr(self, 'prefix_mlp', None))]:
                if module is not None:
                    submodule_prefix = prefix + ('.' if prefix else '') + name
                    for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                        yield m
    
    def zero_grad(self, set_to_none=False):
        """Zero out gradients"""
        if hasattr(self, 'llm'):
            self.llm.zero_grad(set_to_none=set_to_none)
        if hasattr(self, 'protein_encoder'):
            self.protein_encoder.zero_grad(set_to_none=set_to_none)
        if hasattr(self, 'structure_encoder'):
            self.structure_encoder.zero_grad(set_to_none=set_to_none)
        if hasattr(self, 'prefix_mlp'):
            self.prefix_mlp.zero_grad(set_to_none=set_to_none)
    
    def requires_grad_(self, requires_grad=True):
        """Set requires_grad for all parameters"""
        for param in self.parameters():
            param.requires_grad_(requires_grad)
        return self
    
    def get_input_embeddings(self):
        """Get input embeddings from LLM"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'get_input_embeddings'):
            return self.llm.get_input_embeddings()
        return None
    
    def get_output_embeddings(self):
        """Get output embeddings from LLM"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'get_output_embeddings'):
            return self.llm.get_output_embeddings()
        return None
    
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'resize_token_embeddings'):
            return self.llm.resize_token_embeddings(new_num_tokens)
        return None
    
    def tie_weights(self):
        """Tie weights if needed"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'tie_weights'):
            self.llm.tie_weights()
    
    def __call__(self, *args, **kwargs):
        """Make the model callable (delegates to forward)"""
        return self.forward(*args, **kwargs)
    
    def buffers(self, recurse=True):
        """Return all buffers"""
        bufs = []
        if hasattr(self, 'llm'):
            bufs.extend(self.llm.buffers(recurse=recurse))
        if hasattr(self, 'protein_encoder'):
            bufs.extend(self.protein_encoder.buffers(recurse=recurse))
        if hasattr(self, 'structure_encoder'):
            bufs.extend(self.structure_encoder.buffers(recurse=recurse))
        if hasattr(self, 'prefix_mlp'):
            bufs.extend(self.prefix_mlp.buffers(recurse=recurse))
        return iter(bufs)
    
    def named_buffers(self, prefix='', recurse=True, remove_duplicate=True):
        """Return all named buffers"""
        memo = set() if remove_duplicate else None
        for name, module in [('llm', getattr(self, 'llm', None)),
                             ('protein_encoder', getattr(self, 'protein_encoder', None)),
                             ('structure_encoder', getattr(self, 'structure_encoder', None)),
                             ('prefix_mlp', getattr(self, 'prefix_mlp', None))]:
            if module is not None:
                for buffer_name, buffer in module.named_buffers(prefix=f'{prefix}{name}.', recurse=recurse, remove_duplicate=remove_duplicate):
                    if memo is None or buffer not in memo:
                        if memo is not None:
                            memo.add(buffer)
                        yield buffer_name, buffer
    
    def children(self):
        """Return immediate children modules"""
        mods = []
        if hasattr(self, 'llm'):
            mods.append(self.llm)
        if hasattr(self, 'protein_encoder'):
            mods.append(self.protein_encoder)
        if hasattr(self, 'structure_encoder'):
            mods.append(self.structure_encoder)
        if hasattr(self, 'prefix_mlp'):
            mods.append(self.prefix_mlp)
        return iter(mods)
    
    def named_children(self):
        """Return named immediate children modules"""
        for name, module in [('llm', getattr(self, 'llm', None)),
                             ('protein_encoder', getattr(self, 'protein_encoder', None)),
                             ('structure_encoder', getattr(self, 'structure_encoder', None)),
                             ('prefix_mlp', getattr(self, 'prefix_mlp', None))]:
            if module is not None:
                yield name, module
    
    def apply(self, fn):
        """Apply a function to all submodules"""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self
    
    def float(self):
        """Convert to float32"""
        return self.to(torch.float32)
    
    def double(self):
        """Convert to float64"""
        return self.to(torch.float64)
    
    def half(self):
        """Convert to float16"""
        return self.to(torch.float16)
    
    def bfloat16(self):
        """Convert to bfloat16"""
        return self.to(torch.bfloat16)
    
    def type(self, dst_type):
        """Convert to specified dtype"""
        return self.to(dst_type)
    
    def register_buffer(self, name, tensor, persistent=True):
        """Register a buffer (not implemented for wrapper)"""
        raise NotImplementedError("PLLMForCausalLM wrapper doesn't support register_buffer directly")
    
    def register_parameter(self, name, param):
        """Register a parameter (not implemented for wrapper)"""
        raise NotImplementedError("PLLMForCausalLM wrapper doesn't support register_parameter directly")
    
    def add_module(self, name, module):
        """Add a module (not implemented for wrapper)"""
        raise NotImplementedError("PLLMForCausalLM wrapper doesn't support add_module directly")
    
    def get_submodule(self, target):
        """Get a submodule by name"""
        if target == "":
            return self
        atoms = target.split(".")
        mod = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(f"Module has no attribute `{item}`")
            mod = getattr(mod, item)
        return mod
    
    def get_parameter(self, target):
        """Get a parameter by name"""
        module_path, _, param_name = target.rpartition(".")
        mod = self.get_submodule(module_path) if module_path else self
        if not hasattr(mod, param_name):
            raise AttributeError(f"Module has no attribute `{param_name}`")
        param = getattr(mod, param_name)
        if not isinstance(param, torch.nn.Parameter):
            raise AttributeError(f"`{param_name}` is not a Parameter")
        return param
    
    def get_buffer(self, target):
        """Get a buffer by name"""
        module_path, _, buffer_name = target.rpartition(".")
        mod = self.get_submodule(module_path) if module_path else self
        if not hasattr(mod, buffer_name):
            raise AttributeError(f"Module has no attribute `{buffer_name}`")
        buffer = getattr(mod, buffer_name)
        if not isinstance(buffer, torch.Tensor) or isinstance(buffer, torch.nn.Parameter):
            raise AttributeError(f"`{buffer_name}` is not a Buffer")
        return buffer
    
    def _apply(self, fn):
        """Apply function to parameters and buffers"""
        for module in self.children():
            module._apply(fn)
        return self
    
    def _replicate_for_data_parallel(self):
        """Replicate for data parallel (not needed for FSDP)"""
        return self
    
    @property
    def dtype(self):
        """Get dtype of the model"""
        if hasattr(self, 'llm'):
            return next(self.llm.parameters()).dtype
        return torch.float32
    
    @property  
    def device(self):
        """Get device of the model"""
        if hasattr(self, 'llm'):
            return next(self.llm.parameters()).device
        return torch.device('cpu')
    
    @property
    def _no_split_modules(self):
        """Get no-split modules for FSDP wrapping (delegate to underlying LLM)"""
        if hasattr(self, 'llm') and hasattr(self.llm, '_no_split_modules'):
            return self.llm._no_split_modules
        return None
    
    def share_memory(self):
        """Share memory for multiprocessing"""
        for module in self.children():
            module.share_memory()
        return self
    
    def to_empty(self, *, device=None, recurse=True):
        """Move to empty tensors (used by FSDP for memory-efficient initialization)"""
        if recurse:
            for module in self.children():
                if hasattr(module, 'to_empty'):
                    module.to_empty(device=device, recurse=recurse)
        return self
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Save to state dict (used internally by PyTorch)"""
        # Delegate to state_dict()
        pass
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Load from state dict (used internally by PyTorch)"""
        # Delegate to load_state_dict()
        pass
    
    def enable_input_require_grads(self):
        """Enable input gradients (for LoRA)"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'enable_input_require_grads'):
            self.llm.enable_input_require_grads()
        return self
    
    def disable_input_require_grads(self):
        """Disable input gradients"""
        if hasattr(self, 'llm') and hasattr(self.llm, 'disable_input_require_grads'):
            self.llm.disable_input_require_grads()
        return self
    
    def get_fsdp_ignored_modules(self):
        """
        Return list of modules that should be ignored by FSDP.
        These are the frozen encoders that have requires_grad=False.
        """
        ignored = []
        # Only ignore encoders if they're actually frozen (train_encoders=False)
        if hasattr(self, 'train_encoders') and not self.train_encoders:
            if hasattr(self, 'protein_encoder'):
                ignored.append(self.protein_encoder)
            if hasattr(self, 'structure_encoder'):
                ignored.append(self.structure_encoder)
        return ignored


def register_pllm():
    """
    Register PLLM with Transformers' AutoModel system.
    This should be called before training to enable loading PLLM models.
    """
    # Register config
    if "protein_llm_wrapper" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("protein_llm_wrapper", PLLMConfig)
        print("✅ Registered PLLMConfig with AutoConfig")
    
    # Register model
    if PLLMConfig not in MODEL_FOR_CAUSAL_LM_MAPPING:
        MODEL_FOR_CAUSAL_LM_MAPPING.register(PLLMConfig, PLLMForCausalLM)
        print("✅ Registered PLLMForCausalLM with AutoModelForCausalLM")
    
    print("✅ PLLM registration complete!")
    print("   - Encoders are FROZEN (train_encoders=False by default)")
    print("   - Prefix MLP is TRAINABLE (requires_grad=True)")
    print("   - Base LLM is TRAINABLE (requires_grad=True)")


# Auto-register when this module is imported
register_pllm()

