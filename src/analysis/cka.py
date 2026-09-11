"""PRISM V4 — CKA (Centered Kernel Alignment)"""

import torch


def linear_kernel(X, Y=None):
    """Linear kernel: K(X, Y) = X @ Y^T"""
    if Y is None:
        Y = X
    return X @ Y.T


def cka(features_x, features_y):
    """
    计算 linear CKA between two sets of features.

    features_x: [N, D1]
    features_y: [N, D2]

    Returns: scalar CKA value in [0, 1]
    """
    Kx = linear_kernel(features_x)
    Ky = linear_kernel(features_y)

    # HSIC (Hilbert-Schmidt Independence Criterion) with centering
    Kx = _center(Kx)
    Ky = _center(Ky)

    hsic = (Kx * Ky).sum()
    norm = torch.sqrt((Kx * Kx).sum() * (Ky * Ky).sum())

    if norm == 0:
        return torch.tensor(0.0)
    return hsic / norm


def _center(K):
    """Center kernel matrix: K_c = H K H, H = I - 1/n * 11^T"""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - 1.0 / n
    return H @ K @ H


def cka_matrix(layers_x, layers_y):
    """
    计算两组层的 CKA 矩阵。

    layers_x: list of [N, D] tensors (model A's layers)
    layers_y: list of [N, D] tensors (model B's layers)

    Returns: [len(layers_x), len(layers_y)] tensor
    """
    n_x = len(layers_x)
    n_y = len(layers_y)
    matrix = torch.zeros(n_x, n_y)

    for i in range(n_x):
        for j in range(n_y):
            matrix[i, j] = cka(layers_x[i], layers_y[j])

    return matrix


def extract_layer_features(model, dataloader, device, max_batches=10):
    """
    提取模型各层的中间表示。

    返回: list of [N_total, D] tensors, 每个 tensor 是一层的表示
    """
    model.eval()
    features = []
    hooks = []

    def hook_fn(idx):
        def fn(module, input, output):
            # output: [B, N, D] -> collect all positions
            features.append((idx, output.detach().cpu()))
        return fn

    # 注册 hooks
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            h = layer.register_forward_hook(hook_fn(i))
            hooks.append(h)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
            else:
                input_ids = batch[0].to(device)
            model(input_ids)

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 整理为 per-layer [N, D]
    per_layer = {}
    for idx, feat in features:
        # feat: [B, N, D] -> [B*N, D]
        B, N, D = feat.shape
        flat = feat.reshape(B * N, D)
        if idx not in per_layer:
            per_layer[idx] = []
        per_layer[idx].append(flat)

    sorted_features = []
    for i in sorted(per_layer.keys()):
        sorted_features.append(torch.cat(per_layer[i], dim=0))

    return sorted_features
