"""
:file: utils.py
:date: 2026-08-04 (create date) / 2026-08-04 (last modified date)
:description: Utility helpers for vector similarity and norms.
:src: [paper name] Federated Unlearning Compensation
"""

import numpy as np
import torch


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)


def norm(vec, p=2):
    return np.linalg.norm(vec, ord=p)