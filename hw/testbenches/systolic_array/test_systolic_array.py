"""cocotb testbench for hw/rtl/systolic_array.

Run:  pytest hw/testbenches/systolic_array -v
Each test compares the RTL against a numpy reference (the "golden model").
"""
import os
from pathlib import Path

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

N = int(os.getenv("ARRAY_N", "4"))
DW, ACC = 8, 32
RTL = Path(__file__).resolve().parents[2] / "rtl" / "systolic_array"


# ---------------------------------------------------------------- helpers
def pack(vals, width):
    """List of signed ints -> flat bus value (element i at bits [i*width +: width])."""
    out = 0
    for i, v in enumerate(vals):
        out |= (int(v) & ((1 << width) - 1)) << (i * width)
    return out


def unpack(bus, n, width):
    """Flat bus value -> list of signed ints."""
    bus = int(bus)
    vals = []
    for i in range(n):
        v = (bus >> (i * width)) & ((1 << width) - 1)
        if v >= 1 << (width - 1):
            v -= 1 << width
        vals.append(v)
    return vals


async def reset(dut):
    dut.w_load.value = 0
    dut.w_row.value = 0
    dut.w_data.value = 0
    dut.a_valid.value = 0
    dut.a_data.value = 0
    dut.rst_n.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, W):
    """Write the N x N weight matrix, one row per cycle."""
    for r in range(N):
        dut.w_load.value = 1
        dut.w_row.value = r
        dut.w_data.value = pack(W[r], DW)
        await RisingEdge(dut.clk)
    dut.w_load.value = 0


async def run_vectors(dut, A):
    """Stream the rows of A back to back; return the results in order."""
    results = []

    # Results must arrive within the documented latency (2N - 1 cycles) after
    # the last input. Bound the wait so a broken y_valid fails instead of hanging.
    max_cycles = len(A) + 2 * N + 2

    async def collect():
        for _ in range(max_cycles):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if dut.y_valid.value == 1:
                results.append(unpack(dut.y_data.value, N, ACC))
                if len(results) == len(A):
                    return
        raise AssertionError(
            f"got {len(results)} of {len(A)} results within {max_cycles} cycles")

    collector = cocotb.start_soon(collect())
    for row in A:
        dut.a_valid.value = 1
        dut.a_data.value = pack(row, DW)
        await RisingEdge(dut.clk)
    dut.a_valid.value = 0
    await collector
    await RisingEdge(dut.clk)  # leave the read-only phase before driving again
    return np.array(results, dtype=np.int64)


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)


def rand_int8(rng, shape):
    return rng.integers(-128, 128, size=shape, dtype=np.int64)


# ---------------------------------------------------------------- tests
@cocotb.test()
async def test_identity(dut):
    """W = identity, so y must equal a."""
    await start(dut)
    await load_weights(dut, np.eye(N, dtype=np.int64))
    a = np.array([[(-1) ** i * (i + 1) for i in range(N)]], dtype=np.int64)
    y = await run_vectors(dut, a)
    assert (y == a).all(), f"got {y}, expected {a}"


@cocotb.test()
async def test_random_stream(dut):
    """50 random INT8 vectors sent back to back, including -128 extremes."""
    rng = np.random.default_rng(0)
    await start(dut)
    W = rand_int8(rng, (N, N))
    W[0, 0] = -128
    await load_weights(dut, W)
    A = rand_int8(rng, (50, N))
    A[0, :] = -128
    y = await run_vectors(dut, A)
    expected = A @ W
    assert (y == expected).all(), f"mismatch:\n{y}\nvs\n{expected}"


@cocotb.test()
async def test_tiled_dense_layer(dut):
    """A tiny model layer: x (8x13) @ W (13x7) done as N x N tiles, then ReLU.

    This is the pattern a real model uses: the software splits big matrices
    into tiles the array can hold and adds up the partial results. The sizes
    are deliberately not multiples of N, so edge tiles are zero-padded to
    N x N and the extra output columns are trimmed.
    """
    rng = np.random.default_rng(1)
    await start(dut)
    M, K, J = 8, 13, 7
    X = rand_int8(rng, (M, K))
    W = rand_int8(rng, (K, J))
    out = np.zeros((M, J), dtype=np.int64)
    for k in range(0, K, N):           # tile along the shared dimension
        for j in range(0, J, N):       # tile along the output dimension
            w_tile = np.zeros((N, N), dtype=np.int64)
            x_tile = np.zeros((M, N), dtype=np.int64)
            wk, wj = min(N, K - k), min(N, J - j)
            w_tile[:wk, :wj] = W[k:k + wk, j:j + wj]
            x_tile[:, :wk] = X[:, k:k + wk]
            await load_weights(dut, w_tile)
            out[:, j:j + wj] += (await run_vectors(dut, x_tile))[:, :wj]
    out = np.maximum(out, 0)           # ReLU
    expected = np.maximum(X @ W, 0)
    assert (out == expected).all()


# ---------------------------------------------------------------- pytest entry
def test_systolic_array_runner():
    from cocotb_tools.runner import get_runner

    sim = os.getenv("SIM", "icarus")
    runner = get_runner(sim)
    runner.build(
        sources=[RTL / "pe.sv", RTL / "systolic_array.sv"],
        hdl_toplevel="systolic_array",
        parameters={"N": N, "DW": DW, "ACC": ACC},
        build_dir=Path(__file__).parent / "sim_build",
        always=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(hdl_toplevel="systolic_array", test_module="test_systolic_array",
                test_dir=Path(__file__).parent)
