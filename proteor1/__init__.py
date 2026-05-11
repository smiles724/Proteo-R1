"""ProteoR1 - Antibody CDR redesign with text + structure dual-modal models."""

from importlib import import_module

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "ProteoR1UnderstandConfig": ("proteor1.understand", "ProteoR1UnderstandConfig"),
    "ProteoR1UnderstandProcessor": ("proteor1.understand", "ProteoR1UnderstandProcessor"),
    "ProteoR1UnderstandModel": ("proteor1.understand", "ProteoR1UnderstandModel"),
    "ProteoR1GenerateConfig": ("proteor1.generate", "ProteoR1GenerateConfig"),
    "ProteoR1GenerateModel": ("proteor1.generate", "ProteoR1GenerateModel"),
    "Boltz1": ("proteor1.generate", "Boltz1"),
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
