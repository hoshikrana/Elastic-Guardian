"""
EGX — Elastic Guardian X  v0.1.0
Intelligent Adaptive Training Runtime.

Quickstart:
    from egx.api.trainer import EGX
    from egx.api.config import EGXConfig
    result = EGX().train(model=my_model, dataset=train_dataset)

Architecture: 7 layers, 8 DSA structures, 12 inviolable laws.
See: docs/architecture/EGX_Definitive_Architecture.docx
"""

__version__ = "0.1.0"
__author__ = "Hoshik Rana"
__license__ = "Apache-2.0"

# NO imports from egx.api or any other subpackage here.
# Importing Layer 7 (api/) from the package root violates Law 4.
# Import explicitly: from egx.api.trainer import EGX
