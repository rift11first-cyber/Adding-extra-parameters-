"""Structured initialization for the pre-LM-head residual MLP.

A seeded Gaussian draw is converted to a column-orthonormal frame Q with QR.
The hidden projection is W1 = Q and the reconstruction is W2 = alpha * Q^T,
so the linear part of the branch is alpha * I while GELU supplies the
nonlinear residual.  Both biases are zero, making the branch vanish at the
origin and avoiding an uncontrolled constant shift.
"""

import torch

_HIDDEN = 3890
_BRANCH_SCALE = 0.15


def generate_weights(cfg: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """Return deterministic float32 pre-head weights for the given seed."""
    dim = int(cfg["dim"])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sample = torch.randn(_HIDDEN, dim, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(sample, mode="reduced")

    w1 = q.contiguous()
    w2 = (_BRANCH_SCALE * q.t()).contiguous()
    b1 = torch.zeros(_HIDDEN, dtype=torch.float32)
    b2 = torch.zeros(dim, dtype=torch.float32)

    return {
        "pre_head.w1": w1,
        "pre_head.b1": b1,
        "pre_head.w2": w2,
        "pre_head.b2": b2,
    }


if __name__ == "__main__":
    cfg = {
        "dim": 128,
        "hidden_dim": 352,
        "n_layers": 4,
        "vocab_size": 8192,
        "max_seq_len": 1024,
    }
    weights = generate_weights(cfg, seed=0)
    expected = {
        "pre_head.w1": (3890, 128),
        "pre_head.b1": (3890,),
        "pre_head.w2": (128, 3890),
        "pre_head.b2": (128,),
    }

    for name, shape in expected.items():
        tensor = weights[name]
        assert tuple(tensor.shape) == shape
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        print(
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"rms={tensor.square().mean().sqrt().item():.6f} "
            f"min={tensor.min().item():.6f} max={tensor.max().item():.6f}"
        )

    repeated = generate_weights(cfg, seed=0)
    assert all(torch.equal(weights[name], repeated[name]) for name in expected)

    q = weights["pre_head.w1"]
    eye = torch.eye(q.shape[1], dtype=q.dtype)
    orthogonality_error = (q.t() @ q - eye).abs().max().item()

    x = torch.randn(
        4096,
        q.shape[1],
        generator=torch.Generator().manual_seed(123),
        dtype=torch.float32,
    ) * 3.2
    branch = (
        torch.nn.functional.gelu(x @ q.t() + weights["pre_head.b1"])
        @ weights["pre_head.w2"].t()
        + weights["pre_head.b2"]
    )

    print(f"orthogonality_error={orthogonality_error:.3e}")
    print(f"branch_rms_at_input_rms_3.2={branch.square().mean().sqrt().item():.6f}")
    print("self-test passed")
