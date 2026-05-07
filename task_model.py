"""
task_model.py
=============
Thin wrapper that re-exports building blocks and model classes from backbone.py
so that downstream scripts (finetune_classification.py, hard_mining.py, etc.)
can use a single ``import task_model`` to access the full architecture.

All model structures, parameter names, and state-dict keys are defined in
backbone.py — this file adds nothing new to the parameter space.
"""

# Re-export everything a consumer might need from backbone.py.
from backbone import (          # noqa: F401  (these are imported for side-effects / re-export)
    NeuralTransformer,
    NeuralTransformerForMaskedEEGModeling,
    NeuralTransformerForMEM,
    DropPath,
    Mlp,
    Attention,
    Block,
    PatchEmbed,
    TemporalConv,
)
