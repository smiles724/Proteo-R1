"""Data processing helpers for ProteoR1 generation.

The data pipeline has optional parsing dependencies. Keep this package import
lightweight so `from proteor1.generate import ProteoR1GenerateModel` can import
`proteor1.generate.data_load.const` without importing every data type eagerly.
"""

_LAZY_IMPORTS = {
    "Atom": ".types",
    "Bond": ".types",
    "Chain": ".types",
    "Connection": ".types",
    "Interface": ".types",
    "MSA": ".types",
    "MSADeletion": ".types",
    "MSAResidue": ".types",
    "MSASequence": ".types",
    "Residue": ".types",
    "Token": ".types",
    "TokenBond": ".types",
    "Input": ".types",
    "Structure": ".types",
    "Tokenized": ".types",
    "JSONSerializable": ".types",
    "StructureInfo": ".types",
    "AntibodyInfo": ".types",
    "ChainInfo": ".types",
    "InterfaceInfo": ".types",
    "InferenceOptions": ".types",
    "Record": ".types",
    "Manifest": ".types",
    "BoltzTokenizer": ".tokenize",
    "BoltzFeaturizer": ".featurize",
    "pad_dim": ".featurize",
    "pad_to_max": ".featurize",
    "load_input": ".utils",
    "ab_region_type": ".utils",
    "ag_region_type": ".utils",
    "parse_yaml": ".parse",
    "parse_boltz_schema": ".parse",
    "Target": ".parse",
    "Cropper": ".crop",
    "BoltzCropper": ".crop",
    "AntibodyCropper": ".crop",
    "MixedCropper": ".crop",
    "Sample": ".sample",
    "Sampler": ".sample",
    "RandomSampler": ".sample",
    "ClusterSampler": ".sample",
    "AntibodySampler": ".sample",
    "DistillationSampler": ".sample",
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_LAZY_IMPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
