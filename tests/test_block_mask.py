"""Independent block-mask oracle and compiled CUDA forward/backward checks.

Set NFM_REQUIRE_CUDA=1 on the training machine to fail instead of silently
skipping CUDA checks. No slide data or experiment dependencies are required.
"""

import math
import os
from dataclasses import dataclass

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention

from nfm.modeling.block_mask import create_ragged_block_quantized_knn_mask


@dataclass
class MaskCase:
    neighbors: list[torch.Tensor]
    # Explicit validity is test input, never inferred from the production mask.
    valid: list[list[bool]]
    block_size: int
    total: int


def make_case(lengths, *, real=None, holes=(), block_size=128, seed=0):
    """Directed sparse edges, with self edges for every real token."""
    generator = torch.Generator().manual_seed(seed)
    neighbors, validity = [], []
    for doc, length in enumerate(lengths):
        valid = [i < (real[doc] if real is not None else length) for i in range(length)]
        for hole_doc, row in holes:
            if doc == hole_doc:
                valid[row] = False
        real_rows = [i for i, keep in enumerate(valid) if keep]
        table = torch.full((length, 3), -1, dtype=torch.int64)
        for row in real_rows:
            table[row, 0] = row
            # Draw only valid keys, but leave some neighbor slots invalid.
            # Random directed edges expose assumptions of symmetric adjacency.
            if torch.rand((), generator=generator) > 0.25:
                draws = torch.randint(len(real_rows), (2,), generator=generator)
                table[row, 1:] = torch.tensor([real_rows[int(i)] for i in draws])
        neighbors.append(table)
        validity.append(valid)
    return MaskCase(neighbors, validity, block_size, sum(lengths))


def oracle(case):
    """Build the specified block-quantized graph using explicit Python loops.

    Deliberately does not use BlockMask, to_dense(), mask_mod, or the production
    packing/sorting code. One neighbor edge activates a *block pair*, not only
    that token pair; valid same-document pairs within that block pair may attend.
    """
    owners = [-1] * case.total
    active = set()
    offset = 0
    for doc, (table, validity) in enumerate(
        zip(case.neighbors, case.valid, strict=True)
    ):
        for row, valid in enumerate(validity):
            if valid:
                owners[offset + row] = doc
            else:
                assert (table[row] == -1).all()
            for key in table[row].tolist():
                if key >= 0:
                    assert valid and key < len(validity) and validity[key]
                    active.add(
                        (
                            (offset + row) // case.block_size,
                            (offset + key) // case.block_size,
                        )
                    )
        offset += len(validity)
    allowed = torch.tensor(
        [
            [
                owners[q] >= 0
                and owners[q] == owners[k]
                and (q // case.block_size, k // case.block_size) in active
                for k in range(case.total)
            ]
            for q in range(case.total)
        ],
        dtype=torch.bool,
    )
    return torch.tensor(owners), active, allowed


def assert_mask_matches_oracle(case):
    owners, active, expected = oracle(case)
    mask = create_ragged_block_quantized_knn_mask(
        case.neighbors, case.block_size, max_sequence_length=1024
    )
    assert mask.seq_lengths == (case.total, case.total)
    assert mask.mask_mod.doc_ids.shape == (1024,)
    torch.testing.assert_close(
        mask.mask_mod.doc_ids[: case.total].to(torch.int64), owners
    )
    assert (mask.mask_mod.doc_ids[case.total :] == -1).all()
    effective = torch.zeros_like(expected)
    seen = set()
    blocks = math.ceil(case.total / case.block_size)
    for q_block in range(blocks):
        q_start = q_block * case.block_size
        q_end = min(q_start + case.block_size, case.total)
        for counts, indices, full in [
            (mask.kv_num_blocks, mask.kv_indices, False),
            (mask.full_kv_num_blocks, mask.full_kv_indices, True),
        ]:
            count = int(counts[0, 0, q_block])
            row = indices[0, 0, q_block]
            assert 0 <= count <= len(row)
            assert (row[count:] == -1).all()
            for k_block in row[:count].tolist():
                assert 0 <= k_block < blocks
                pair = (q_block, k_block)
                assert pair not in seen, f"Duplicate full/partial block: {pair}"
                seen.add(pair)
                k_start = k_block * case.block_size
                k_end = min(k_start + case.block_size, case.total)
                if full:
                    # Full blocks bypass mask_mod, including its padding check.
                    assert q_end - q_start == case.block_size
                    assert k_end - k_start == case.block_size
                    assert expected[q_start:q_end, k_start:k_end].all()
                    effective[q_start:q_end, k_start:k_end] = True
                else:
                    q = torch.arange(q_start, q_end)[:, None]
                    k = torch.arange(k_start, k_end)[None, :]
                    effective[q_start:q_end, k_start:k_end] = mask.mask_mod(0, 0, q, k)
    assert seen == active, f"Missing {active - seen}; extra {seen - active}"
    torch.testing.assert_close(effective, expected, atol=0, rtol=0)
    return mask, owners, expected


BOUNDARY_CASES = [
    pytest.param({"lengths": [128, 256]}, id="aligned-documents"),
    pytest.param({"lengths": [256, 128], "real": [129, 1]}, id="per-crop-padding"),
    pytest.param({"lengths": [73, 139, 91]}, id="unaligned-document-boundaries"),
    pytest.param({"lengths": [1, 127, 128, 129]}, id="block-boundary-lengths"),
    pytest.param({"lengths": [150, 129]}, id="incomplete-final-block"),
    pytest.param(
        {
            "lengths": [256, 128],
            "holes": [(0, 1), (0, 63), (0, 127), (0, 128), (0, 200)],
        },
        id="internal-padding",
    ),
    pytest.param(
        {"lengths": [128, 128], "real": [128, 0]}, id="entirely-padded-document"
    ),
    pytest.param({"lengths": [1]}, id="single-nucleus"),
    pytest.param({"lengths": [128], "real": [0]}, id="all-rows-masked"),
]


@pytest.mark.parametrize("settings", BOUNDARY_CASES)
def test_mask_structure_against_independent_oracle(settings):
    assert_mask_matches_oracle(make_case(**settings))


@pytest.mark.parametrize("seed", range(32))
def test_randomized_mask_structure(seed):
    generator = torch.Generator().manual_seed(seed)
    block_size = [4, 8, 16][seed % 3]
    lengths = torch.randint(1, 35, (1 + seed % 4,), generator=generator).tolist()
    holes = [
        (doc, row)
        for doc, length in enumerate(lengths)
        for row in range(length)
        if torch.rand((), generator=generator) < 0.2
    ]
    case = make_case(
        lengths,
        block_size=block_size,
        holes=holes,
        seed=seed,
    )
    assert_mask_matches_oracle(case)


def test_fixed_doc_ids_capacity_with_variable_lengths_and_partial_tiles():
    capacity, block_size = 257, 128
    masks = []
    for length in (1, 127, 128, 129, 255, capacity):
        mask = create_ragged_block_quantized_knn_mask(
            [torch.arange(length)[:, None]],
            block_size,
            max_sequence_length=capacity,
        )
        assert mask.mask_mod.doc_ids.shape == (capacity,)
        assert mask.seq_lengths == (length, length)
        assert mask.kv_num_blocks.shape[-1] == math.ceil(length / block_size)
        # Include all positions in the final tile, even beyond doc_ids capacity.
        indices = torch.arange(math.ceil(length / block_size) * block_size)
        actual = mask.mask_mod(0, 0, indices[:, None], indices[None, :])
        expected = (indices[:, None] < length) & (indices[None, :] < length)
        torch.testing.assert_close(actual, expected)
        masks.append(mask)
    # Later batches must not change document IDs held for a pending backward.
    for length, mask in zip((1, 127, 128, 129, 255, capacity), masks, strict=True):
        assert (mask.mask_mod.doc_ids[:length] == 0).all()
        assert (mask.mask_mod.doc_ids[length:] == -1).all()


@pytest.fixture
def compiled_cuda_attention():
    if not torch.cuda.is_available():
        if os.environ.get("NFM_REQUIRE_CUDA") == "1":
            pytest.fail("NFM_REQUIRE_CUDA=1 but CUDA is unavailable")
        pytest.skip("Requires a CUDA GPU; set NFM_REQUIRE_CUDA=1 to require execution")
    # Cases deliberately change shape, dtype, and mask metadata. Clear guards
    # between tests so compiler cache limits cannot masquerade as mask failures.
    torch.compiler.reset()
    # Compile the actual FlexAttention kernel, not its eager reference path.
    return torch.compile(
        flex_attention,
        fullgraph=True,
        dynamic=True,
        options={"triton.cudagraphs": False},
    )


def dense_attention_oracle(q, k, v, allowed):
    # FP64 makes the oracle independent of fused kernels and TF32 arithmetic.
    logits = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    allowed = allowed.to(q.device)
    logits = logits.masked_fill(~allowed, -torch.inf)
    # Entirely masked rows have zero output and zero Q/K/V contributions.
    logits = torch.where(allowed.any(dim=-1, keepdim=True), logits, 0.0)
    probability = logits.softmax(dim=-1).masked_fill(~allowed, 0.0)
    return probability @ v.double()


def test_dense_oracle_has_expected_values_and_masked_gradients():
    q = torch.zeros(1, 1, 3, 2, requires_grad=True)
    k = torch.ones_like(q, requires_grad=True)
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]], requires_grad=True)
    allowed = torch.tensor(
        [[True, True, False], [False, False, False], [False, False, True]]
    )
    output = dense_attention_oracle(q, k, v, allowed)
    torch.testing.assert_close(
        output,
        torch.tensor([[[[2.0, 3.0], [0.0, 0.0], [5.0, 6.0]]]], dtype=torch.float64),
    )
    output.sum().backward()
    torch.testing.assert_close(
        v.grad, torch.tensor([[[[0.5, 0.5], [0.5, 0.5], [1.0, 1.0]]]])
    )
    assert torch.isfinite(q.grad).all() and (q.grad[:, :, 1] == 0).all()
    assert (k.grad == 0).all()


def assert_cuda_matches_reference(mask, owners, reference, total, dtype, compiled):
    generator = torch.Generator(device="cuda").manual_seed(41)
    shape = (2, 2, total, 32)
    inputs = [
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
        for _ in range(3)
    ]
    upstream = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    results, gradients = [], []
    for use_flex in [True, False]:
        q, k, v = [x.detach().clone().requires_grad_() for x in inputs]
        if use_flex:
            output = compiled(q, k, v, block_mask=mask.to("cuda"))
        else:
            output = reference(q, k, v)
        # Nonuniform upstream gradients test more than a symmetric mean loss.
        (output.double() * upstream.double()).sum().backward()
        torch.cuda.synchronize()
        results.append(output.detach().double())
        gradients.append([x.grad.detach().double() for x in (q, k, v)])
    atol, rtol = (2e-5, 3e-4) if dtype == torch.float32 else (0.02, 0.03)
    torch.testing.assert_close(results[0], results[1], atol=atol, rtol=rtol)
    for name, actual, expected in zip("QKV", gradients[0], gradients[1], strict=True):
        assert torch.isfinite(actual).all(), f"Nonfinite {name} gradient"
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda msg, name=name: f"{name} gradient: {msg}",
        )
    padding = (owners < 0).to("cuda")
    assert torch.isfinite(results[0]).all()
    assert (results[0][:, :, padding] == 0).all()
    for gradient in gradients[0]:
        assert (gradient[:, :, padding] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("settings", BOUNDARY_CASES)
def test_cuda_forward_and_qkv_gradients(settings, dtype, compiled_cuda_attention):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.fail("This validation suite requires a CUDA GPU with BF16 support")
    case = make_case(**settings)
    mask, owners, allowed = assert_mask_matches_oracle(case)
    assert_cuda_matches_reference(
        mask,
        owners,
        lambda q, k, v: dense_attention_oracle(q, k, v, allowed),
        case.total,
        dtype,
        compiled_cuda_attention,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_cuda_training_token_budget(dtype, compiled_cuda_attention):
    """Exercise 65,536 positions without allocating an N-by-N dense oracle."""
    total, block_size, real = 65536, 128, 65519
    # Self edges select exactly the diagonal block pairs. Within each selected
    # block attention remains dense over real tokens (block-quantized semantics).
    neighbors = torch.full((total, 1), -1, dtype=torch.int64)
    neighbors[:real, 0] = torch.arange(real)
    mask = create_ragged_block_quantized_knn_mask(
        [neighbors], block_size, max_sequence_length=total
    )
    owners = torch.full((total,), -1, dtype=torch.int64)
    owners[:real] = 0
    valid = (torch.arange(total) < real).view(-1, block_size)
    allowed = valid[:, :, None] & valid[:, None, :]

    def reference(q, k, v):
        # A batch of independent dense 128-by-128 problems is the exact oracle
        # for this graph; it uses O(N * block_size), rather than O(N**2), memory.
        reshaped = [
            x.unflatten(-2, (total // block_size, block_size)) for x in (q, k, v)
        ]
        return dense_attention_oracle(*reshaped, allowed).reshape(q.shape)

    assert_cuda_matches_reference(
        mask, owners, reference, total, dtype, compiled_cuda_attention
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_cuda_padding_and_other_documents_cannot_affect_crop(
    dtype, compiled_cuda_attention
):
    case = make_case([256, 128], real=[150, 100])
    mask, owners, _ = assert_mask_matches_oracle(case)
    mask = mask.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(52)
    inputs = [
        torch.randn(
            (1, 2, case.total, 32), device="cuda", dtype=dtype, generator=generator
        )
        for _ in range(3)
    ]
    keep = (owners == 0).to("cuda")
    with torch.no_grad():
        original = compiled_cuda_attention(*inputs, block_mask=mask)
        changed = [x.clone() for x in inputs]
        for x in changed:
            x[:, :, ~keep] = (
                torch.randn(
                    x[:, :, ~keep].shape,
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )
                * 20
            )
        perturbed = compiled_cuda_attention(*changed, block_mask=mask)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        original[:, :, keep], perturbed[:, :, keep], atol=0, rtol=0
    )
