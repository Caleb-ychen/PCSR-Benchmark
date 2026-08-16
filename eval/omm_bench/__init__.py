"""OMM-Bench unified inference library.

Refactored from bench_v12_hotfix15_py: all 30 model scripts share the same
prompt template, IO utilities, and main loop; only the model-loading and
single-item inference step differ, so those are isolated into backends/.
"""
__version__ = "0.1.0"
