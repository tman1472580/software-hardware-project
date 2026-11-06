# AI Inference Chip Architectures: TPU vs NPU vs GPU

**Purpose:** understand how the three main families of AI chips are built, then use that to pick the design for *this* project's FPGA accelerator (INT4 small language models at the edge).

**Confidence tags:** **[Certain]** = published spec or paper. **[Likely]** = strong inference or widely reported figure. **[Guessing]** = assumption to be checked.

---

## 0. The one thing to know first

**For language model inference at batch size 1, every one of these chips spends most of its time waiting on memory, not doing math.** [Certain]

Generating one token ("decode") reads *every weight in the model once* and does only one multiply-add per weight. With INT4 weights that is 2 operations per half-byte, so **4 operations per byte of memory traffic** [Certain, arithmetic]. Every chip below can do hundreds to thousands of operations per byte. So the size of your multiplier array mostly decides how fast you read the prompt ("prefill"), while **memory bandwidth decides how fast you generate tokens**.

Keep this in mind through the whole doc. It is the main design constraint for the FPGA in section 5.

---

## 1. The shared building blocks

Every AI chip is a mix of the same four parts. They differ in how much area each part gets and who controls it.

| Part | Job | Why it matters |
|---|---|---|
| **Compute fabric** | Do the multiply-adds (matrix math) | Sets peak speed |
| **On-chip memory** | Hold data close to compute (SRAM) | Each off-chip byte costs ~100x the energy of a multiply-add [Likely] |
| **Off-chip memory** | Hold the model (DRAM / HBM) | Sets token generation speed |
| **Control** | Decide what runs when | Hardware (GPU) or compiler (TPU/NPU) |

### Dataflow: which number stays put

A matrix multiply `C = A × B` reuses each input many times. The "dataflow" is the choice of which value sits still inside a processing element (PE) while the others move past it.

| Dataflow | What stays in the PE | Used by | Good when |
|---|---|---|---|
| **Weight-stationary** | One weight | Google TPU [Certain] | Many inputs share the same weights (big batches, prefill) |
| **Output-stationary** | One partial sum | Many NPUs, ShiDianNao [Likely] | Long dot products, avoid moving partial sums |
| **Input-stationary** | One activation | Rare | Few weights, many outputs |
| **Row-stationary** | A row of a convolution | MIT Eyeriss [Certain] | Convolutions (CNNs), energy-optimised |

For transformers, the operations are almost all plain matrix multiplies, so the CNN-specific dataflows (row-stationary) bring little. Weight-stationary and output-stationary are the real contenders.

---

## 2. TPU: one giant systolic array, controlled by the compiler

### 2.1 How it is built

The original TPU (v1, 2015, published 2017) is the cleanest reference design. [Certain, Jouppi et al., ISCA 2017]

- **Compute:** a 256 × 256 grid of 8-bit multiply-add units (65,536 MACs) running at 700 MHz = 92 TOPS.
- **Systolic:** data pulses through the grid like a heartbeat. Activations flow left to right, partial sums flow top to bottom. Each PE only talks to its neighbours, so there are no long wires and no shared buses. That is why it can be so dense.
- **Weight-stationary:** each PE holds one weight. Weights are loaded from a FIFO, and there is a second "shadow" weight register so the next tile loads while the current one computes (double buffering).
- **On-chip memory:** 24 MiB "Unified Buffer" for activations, 4 MiB of 32-bit accumulators below the array.
- **Off-chip:** DDR3 at 34 GB/s. The paper itself shows most of their production models were limited by this bandwidth, not by the array. [Certain]
- **Control:** a host CPU sends ~a dozen big CISC instructions: `Read_Host_Memory`, `Read_Weights`, `MatrixMultiply`, `Activate`, `Write_Host_Memory`. No caches, no branch prediction, no threads. Timing is deterministic, so the compiler plans everything. [Certain]
- **Activation unit:** fixed hardware after the accumulators for ReLU, sigmoid, tanh, pooling.

```text
             weights (loaded from top, then held)
               |    |    |    |
 activations → [PE]→[PE]→[PE]→[PE]
 (from left)   [PE]→[PE]→[PE]→[PE]      each PE: acc_out = acc_in + a × w
               [PE]→[PE]→[PE]→[PE]                pass a to the right
               [PE]→[PE]→[PE]→[PE]                pass acc down
                 ↓    ↓    ↓    ↓
              accumulators → activation unit → unified buffer
```

### 2.2 How it evolved

- **v2/v3:** added HBM memory and bfloat16 for training, 128 × 128 arrays (several per chip), chip-to-chip links. [Likely]
- **v4 and later:** multiple 128 × 128 matrix units per core, HBM at ~1.2 TB/s, thousands of chips joined by optical switches in a 3D torus. [Likely]

The trend matters: Google went **from one 256 × 256 array to several 128 × 128 arrays**. A huge array wastes most of its PEs when a matrix dimension is smaller than the array. [Likely]

### 2.3 Strengths and weaknesses for inference

- ✅ Very high math per mm² and per watt. Simple, regular hardware.
- ✅ Deterministic timing. Easy to simulate and verify, which is exactly what this project needs.
- ❌ Poor at anything that is not a big matrix multiply (softmax, layer norm, sorting). Needs a separate vector unit, which later TPUs added.
- ❌ Utilization collapses at batch size 1, because each loaded weight is used once.

---

## 3. GPU: thousands of small cores, controlled by hardware

### 3.1 How it is built (NVIDIA as the reference)

- **Streaming Multiprocessors (SMs):** the GPU is ~100+ SMs (H100: 132 in the SXM part). [Likely] Each SM runs many threads.
- **SIMT:** threads run in groups of 32 called *warps*. All 32 run the same instruction on different data. When one warp waits on memory, the hardware instantly switches to another. **This is how GPUs hide memory latency: by having many threads ready, not by being clever about data.** [Certain]
- **Tensor Cores:** small fixed matrix-multiply units inside each SM (e.g., a 4 × 4 or larger tile per cycle). They are essentially mini systolic/dot-product arrays bolted onto a general processor. They do most of the AI math. [Certain]
- **Memory hierarchy:** big register file (256 KB per SM), shared memory/L1 (up to ~228 KB per SM on H100), large L2 (50 MB on H100), then HBM (3.35 TB/s on H100 SXM). [Likely]
- **Control:** hardware schedulers, caches, and a general instruction set. Software (CUDA) can express any algorithm.

### 3.2 How GPUs make inference fast

None of these is hardware magic; they are software tricks that work around the memory wall:

- **Batching:** serve many users at once, so each weight read from memory is reused many times. Turns a bandwidth problem into a compute problem.
- **FlashAttention:** compute attention in tiles that fit in shared memory, with a running ("online") softmax, so the big attention matrix never goes to HBM. [Certain]
- **Paged KV cache:** store each user's attention history in fixed-size blocks, like virtual memory, to avoid wasted memory. [Certain]
- **Low precision:** FP8 / INT8 / INT4 weights cut the bytes read per token.

### 3.3 Strengths and weaknesses for inference

- ✅ Runs anything. New model types work on day one.
- ✅ Huge memory bandwidth (HBM).
- ❌ A lot of area and power goes to scheduling, caches and register files instead of math. [Likely]
- ❌ At batch size 1 it is also memory-bound. A datacenter GPU generating one stream of tokens uses a small fraction of its compute.
- ❌ SIMT does not map well onto an FPGA. Copy the software tricks, not the hardware.

---

## 4. NPU: small, fixed-function, power-first

"NPU" is a loose label for the AI block in phones, laptops and microcontrollers. Examples: Apple Neural Engine, Qualcomm Hexagon NPU, Arm Ethos-U55/U65, Google Edge TPU. [Certain that these exist; internal details are mostly unpublished, so the details below are **[Likely]**.]

### 4.1 How they are built

- **Compute:** a modest MAC array (Ethos-U55: 32 to 256 MACs per cycle [Likely]), usually INT8, increasingly INT4. Often output-stationary.
- **Separate vector/activation unit:** fixed hardware or a small SIMD unit for activations, pooling, softmax, normalisation. Qualcomm's design is explicitly split into scalar, vector and tensor units. [Likely]
- **Scratchpad, not cache:** on-chip SRAM is managed explicitly by the compiler. A DMA engine moves tiles in and out, double-buffered so loading the next tile overlaps compute.
- **Shared system memory:** no dedicated DRAM. The NPU shares LPDDR with the CPU and GPU, so bandwidth is tens of GB/s, not TB/s.
- **Weight compression:** weights are stored compressed in DRAM and decompressed on the fly at the array edge. This directly raises effective bandwidth.
- **Compiler-driven:** a vendor compiler turns a model graph into a static command stream. Unsupported operations fall back to the CPU, which is slow and is where most NPU performance problems come from.

### 4.2 Strengths and weaknesses for inference

- ✅ Best performance per watt. Built for batch size 1.
- ✅ Designed around the same constraint as this project: little memory bandwidth, shared with a CPU.
- ❌ Inflexible. Anything the compiler does not support runs on the CPU.
- ❌ Most were designed for CNNs. Transformer support (attention, large KV caches) was added later and is often weak.

---

## 5. Side-by-side

| | TPU | GPU | NPU |
|---|---|---|---|
| Compute fabric | Large systolic arrays | Many SIMT cores + tensor cores | Small MAC array + vector unit |
| Dataflow | Weight-stationary | Software-chosen tiling | Mostly output-stationary |
| Who schedules | Compiler | Hardware + CUDA | Compiler |
| On-chip memory | Large scratchpad | Registers + caches | Scratchpad + DMA |
| Off-chip memory | HBM (dedicated) | HBM (dedicated) | Shared LPDDR |
| Precision | INT8, BF16, FP8 | FP16 down to FP4 | INT8, INT4 |
| Best at | Big-batch serving | Flexibility, any model | Batch-1, low power |
| Batch-1 LLM decode | Memory-bound | Memory-bound | Memory-bound |
| Closest to this project | **Compute array** | Kernel techniques | **System design** |

**Takeaway:** this project's README calls the accelerator "TPU-style," but its actual situation (FPGA, shared DDR, batch size 1, INT4, power-limited) is an **NPU**. The right design is a TPU-style array inside an NPU-style system, running GPU-style kernels (tiling, FlashAttention).

---

## 6. Recommended design for this project

### 6.1 Assumptions (check these)

- **Board:** a Zynq UltraScale+ class part such as the AMD Kria KV260. [Guessing: no board is named in the repo yet.] Roughly: ~1,250 DSP slices, ~3 MB of on-chip RAM, one 64-bit DDR4-2400 channel = 19.2 GB/s peak, maybe ~13 GB/s in practice. [Likely for the specs, Guessing for the 70% efficiency]
- **Model:** a ~1B parameter model such as TinyLlama-1.1B (22 layers, hidden size 2048, grouped-query attention with 4 KV heads). [Guessing on model choice, Likely on its shape]
- **Precision:** INT4 weights, INT8 activations, INT32 accumulation ("W4A8"). The README fixes INT4 weights and INT32 accumulation; INT8 activations is my assumption.

### 6.2 The numbers that drive the design

| Quantity | Value | Confidence |
|---|---|---|
| Weight bytes per token (1.1B × 0.5 B) | ~0.55 GB | Likely |
| **Decode ceiling** at ~13 GB/s | **~24 tokens/s** for 1.1B, ~50 tokens/s for 0.5B | Likely |
| Math needed to keep up with DDR in decode | ~26 G MAC/s | Likely |
| 16 × 16 array at 200 MHz | 51 G MAC/s | Certain (arithmetic) |
| 32 × 32 array at 250 MHz | 256 G MAC/s | Certain (arithmetic) |
| Prefill of a 512-token prompt, 1.1B | ~0.56 T MACs: ~11 s on 16 × 16, ~2.2 s on 32 × 32 | Likely |
| KV cache, 2048 tokens, INT8 | ~23 MB: lives in DDR, not on chip | Likely |
| All weights on chip? | No: 0.55 GB vs ~3 MB. Weights must stream from DDR. | Certain |

What this means:

1. **A bigger array does not make token generation faster.** Even 16 × 16 has twice the math decode can use. [Likely]
2. **Array size is a prefill (time-to-first-token) decision.** 32 × 32 is the sweet spot for this class of board. Packing two INT4 multiplies into one DSP slice (a known trick [Likely]) brings that to ~512 DSPs, which fits.
3. **The real decode bottleneck is how fast weights get from DDR into the multipliers.** A textbook systolic array cannot take them fast enough at batch size 1. This is the part most student TPU designs get wrong. See 6.3.

### 6.3 The trap in a textbook weight-stationary array

In a classic TPU array, weights shift in from the top, one row per cycle. Loading a 32 × 32 tile takes 32 cycles and delivers 32 weights per cycle = 16 bytes/cycle. At 250 MHz that is **4 GB/s**, a third of what the DDR can supply. [Likely]

The obvious fix, widening the load path, **does not work on its own**. The problem is *when* each PE can switch to its new weight.

**Cycle schedule for one tile (N = 32, M = number of tokens in the batch):**

- Activations enter skewed: row `r` gets its first value at cycle `r`, then it moves one column to the right each cycle. So PE `(r, c)` first sees tile data at cycle `r + c` and last sees it at cycle `r + c + M - 1`. [Certain, standard systolic timing]
- The "use the new weight" signal must travel with the first activation of the next tile, so PE `(r, c)` switches at cycle `r + c + M`. The front of each tile is a diagonal wave that takes `2N - 1 = 63` cycles to cross the array.
- With one shadow register per PE, PE `(r, c)` can only accept tile `k+2` after it switched to tile `k+1`. So the shadow for a PE is free for exactly `M` cycles, and those windows are staggered diagonally across the array.

What that gives:

| Phase | M | Shadow free window per PE | Load needed | Tile period | Result |
|---|---|---|---|---|---|
| Prefill, 512-token prompt | 512 | 512 cycles | 32 cycles (1 row/cycle) | 512 cycles | ✅ Textbook design is fine. Fill/drain overhead is 63/512 ≈ 12%. |
| Prefill, short prompt | 64 | 64 cycles | 32 cycles | 64 cycles | ✅ Still fine. |
| **Decode** | **1** | **1 cycle** | 8 cycles even at 4 rows/cycle | ≥ 32 cycles | ❌ About 1,024 weights per 32 cycles = 32 weights/cycle. Same 4 GB/s limit. |

At M = 1 each PE's free window is one cycle, and it moves diagonally. A wider row-at-a-time load bus cannot hit a diagonal one-cycle window for all 32 PEs in a row. Making it work needs about `63 / 8 ≈ 8` shadow weight sets per PE plus weights stored in DDR in diagonal order. That is a lot of registers and a hard-to-verify design. [Likely]

**Fix: give decode its own simple datapath.** Do not run batch-1 decode through the systolic array at all.

- **GEMV engine:** 128 INT4 × INT8 multipliers feeding an adder tree. No systolic movement, so no switch hazard.
- **Schedule:** each cycle, the DMA delivers 128 weights (64 bytes) from one row of the weight matrix. The matching 128 activations are already held in registers. One cycle = 128 MACs into one output's running sum. For a 2048 × 2048 matrix: `2048 / 128 = 16` cycles per output, × 2048 outputs = 32,768 cycles for 4.2 M weights = **128 weights/cycle, 16 GB/s at 250 MHz**. That is above the ~13 GB/s DDR delivers, so DDR stays the limit, which is what you want. [Likely, arithmetic]
- **Cost:** about 64 DSP slices with INT4 packing, small next to the ~512 for the array. [Likely]
- **Dequantise and unpack INT4 at the engine input**, not in DDR, so memory traffic stays at half a byte per weight.

The systolic array stays textbook: one shadow register, one-row-per-cycle load. It only runs prefill (and batched decode if you ever add it, once M ≥ 32).

**This schedule is a paper estimate.** The simulator in 6.6 step 1 should reproduce both tile periods before any RTL is written.

### 6.4 Proposed block diagram (v0.1)

```text
        ┌──────────────────────── Zynq PS (ARM A53) ────────────────────────┐
        │  runs the engine (sw/engine), tokenizer, writes command queue      │
        └──────────────┬───────────────────────────────────────┬────────────┘
                       │ AXI-Lite (control regs)               │ shared DDR4 (weights, KV, cmds)
        ┌──────────────▼───────────────────────────────────────▼────────────┐
        │  Command processor  ──►  DMA engine (2 channels, double-buffered)   │
        │        │                     │ weights (INT4)      │ activations    │
        │        │              ┌──────▼──────┐        ┌─────▼──────┐         │
        │        │              │ INT4 unpack │        │ Act buffer │ (BRAM)  │
        │        │              │ + dequant   │        │  INT8      │         │
        │        │              └──────┬──────┘        └─────┬──────┘         │
        │        │                ┌────▼──────────────────────▼───┐           │
        │        └──────────────► │ PREFILL: 32×32 weight-stationary│           │
        │                         │   systolic array               │           │
        │                         │ DECODE: 128-lane GEMV engine   │           │
        │                         │ both INT4 × INT8 → INT32       │           │
        │                         └────────────┬───────────────────┘           │
        │                         ┌────────────▼───────────────────┐           │
        │                         │ Accumulators + per-group scale │ (URAM)    │
        │                         └────────────┬───────────────────┘           │
        │                         ┌────────────▼───────────────────┐           │
        │                         │ Vector unit, 32 lanes           │           │
        │                         │ RMSNorm, softmax, SiLU, RoPE,   │           │
        │                         │ requantise to INT8              │           │
        │                         └─────────────────────────────────┘           │
        └──────────────────────────────── FPGA fabric (PL) ─────────────────────┘
```

Design choices and why:

| Choice | Borrowed from | Reason |
|---|---|---|
| Weight-stationary systolic array, 32 × 32 | TPU | Simplest dense compute; deterministic, easy to simulate |
| Separate 128-lane GEMV engine for decode | NPU-style specialisation | Section 6.3: a systolic array cannot take weights fast enough at batch size 1 |
| Vector unit on the FPGA | NPU, later TPUs | Without it, every layer bounces to the ARM for norm/softmax: ~22 layers × several ops per token of round trips and cache flushes |
| Scratchpad + DMA, no caches | NPU | Predictable timing; the compiler/engine plans all data movement |
| Shared DDR with the ARM, AXI not PCIe | NPU | On a Zynq board the CPU is on the same chip. PCIe (as the README mentions) only applies if you use a separate host PC. [Likely] |
| Few big CISC instructions | TPU | Keeps the decoder tiny; ISA is the contract between hw/ and sw/ |
| Tiled attention with online softmax | GPU (FlashAttention) | Attention matrix never leaves the chip |
| Per-group INT4 scales applied at the accumulator | GPU/NPU quantisation practice | Better accuracy than one scale per row, costs one multiplier per column |

### 6.5 Draft instruction set (starting point for `bridge/isa/`)

| Instruction | What it does |
|---|---|
| `DMA_LOAD dst, src, len` | DDR → on-chip buffer |
| `DMA_STORE dst, src, len` | On-chip buffer → DDR |
| `LOAD_W tile` | Weight buffer → shadow weight registers |
| `MATMUL act, acc, rows, accumulate` | Prefill: stream activations through the array, write or add to accumulators |
| `GEMV w_src, act, acc, rows, cols` | Decode: stream weights from DMA through the GEMV engine |
| `VEC op, src, dst` | Vector unit: norm, softmax, SiLU, RoPE, add, requantise |
| `SYNC` / `FENCE` | Wait for DMA or compute to finish |
| `HALT` | Signal done to the ARM |

### 6.6 Build order (de-risk first)

1. **Simulator before RTL.** A cycle-approximate Python model of the array + DMA in `bridge/simulator/`. Use it to confirm the decode ceiling and the tile periods in 6.3 before writing any Verilog.
2. **4 × 4 array in RTL** with a testbench, checked against the simulator bit for bit.
3. Scale to 32 × 32. Add the GEMV engine for decode.
4. DMA + command processor, then the vector unit.
5. On board: the 2 × 2 "hello world" from the README, then one transformer layer, then the full model.

---

## 7. Open questions

- Which board? Everything in section 6 depends on DDR bandwidth and DSP count.
- Which model and context length? Sets bytes per token and KV cache size.
- Is time-to-first-token or tokens/second the goal? That decides how much area goes to the array versus the memory path.
- W4A8 or W4A16? INT8 activations are cheaper in hardware but cost some accuracy.

## References

- Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit*, ISCA 2017.
- Chen et al., *Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs*, ISCA 2016.
- Kung, *Why Systolic Architectures?*, IEEE Computer 1982.
- Dao et al., *FlashAttention*, NeurIPS 2022.
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM), SOSP 2023.
- Williams et al., *Roofline: An Insightful Visual Performance Model*, CACM 2009.
