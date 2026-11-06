# Getting Started: Simulating Your Own AI Accelerator

**Goal:** write the accelerator in an HDL, test it in simulation, and run small TinyML models on it before touching an FPGA.

**Confidence tags:** **[Certain]** = tested in this repo or a published fact. **[Likely]** = strong inference. **[Guessing]** = assumption.

---

## 0. Three decisions before you install anything

### 0.1 Build one chip, not three

I disagree with building an NPU, a TPU *and* a GPU. A GPU in RTL means thread schedulers, register files, caches and a compiler. That is a multi-year project on its own, and it is the worst fit for batch-1 inference on an FPGA (see `architecture-study.md`). [Likely]

**Build one NPU-style accelerator with a systolic array at its core.** That is the TPU idea inside an NPU system, which is exactly what this repo already plans. You learn all the important parts (dataflow, memory, ISA, quantisation) without the GPU overhead. A tiny SIMT core can be a side project later.

### 0.2 Use SystemVerilog, not VHDL

| | SystemVerilog | VHDL |
|---|---|---|
| Free simulators | Icarus, Verilator (fast) | GHDL only |
| Free synthesis (Yosys) | Yes | Only via a plugin |
| Used by AI chip companies | Most of them [Likely] | Mostly defence/aerospace/Europe |
| Open-source accelerator examples | Many (Gemmini, NVDLA, TinyTPU projects) | Few |

**Use synthesisable SystemVerilog** (the subset that works in both Icarus and Verilator: `logic`, `always_ff`, `always_comb`, `generate`, parameters). Do not use SV classes or UVM. Do the testbench in Python instead (next point). [Likely]

### 0.3 Write testbenches in Python, not in HDL

With **cocotb**, the testbench is Python. That means numpy computes the correct answer, and later the same Python code loads a real TinyML model and compares every layer. A hand-written HDL testbench cannot do that easily. [Certain]

### A word on "emulation"

- **Simulation**: software runs your RTL cycle by cycle (Icarus, Verilator). This is what you want now.
- **Emulation / FPGA prototyping**: your RTL runs on real FPGA hardware. That comes at the end (step 7).

Simulation is slow. Expect thousands to millions of cycles per second depending on the simulator and design size. That is fine for TinyML models and far too slow for language models, which is why TinyML is the right place to start. [Likely]

---

## 1. Software list

### Must have (step 1 onwards)

| Tool | What it does | Install |
|---|---|---|
| **Icarus Verilog 12** | Simple simulator. Start here, it just works with cocotb. | `apt install iverilog` / `brew install icarus-verilog` |
| **cocotb 2.x** | Python testbenches | `pip install -r requirements-dev.txt` |
| **numpy, pytest** | Golden-model maths, test runner | same file |
| **GTKWave** or **Surfer** | View waveforms when a test fails | `apt install gtkwave` / surfer-project.org |
| **Verilator** (lint only, at first) | Catches width mismatches and unused signals that Icarus ignores | `apt install verilator` |
| **Git + VS Code** with the "Verilog-HDL/SystemVerilog" extension | Editing | — |

### Add later

| Tool | When | Why |
|---|---|---|
| **Verilator ≥ 5.036** as the simulator | Step 6, when Icarus gets slow | ~10 to 100× faster than Icarus [Likely]. **Ubuntu 24.04's packaged Verilator (5.020) does not work with cocotb 2.x. I tested it and the build fails.** [Certain] Get a newer one from the OSS CAD Suite or build it from source. |
| **Yosys** | Step 5 | Checks your RTL is synthesisable and gives a rough size before you have an FPGA |
| **SymbiYosys** | Optional | Formal proofs for small blocks (FIFOs, arbiters) |
| **TensorFlow + LiteRT** (formerly TFLite) or **PyTorch** | Step 2 | Train and quantise models to INT8 |
| **Netron** | Step 2 | Look inside a `.tflite` / `.onnx` file to see layers, shapes and scales |
| **AMD Vivado** (free edition) | Step 7 | Synthesis and bitstreams for Zynq / Kria boards |

**Easiest install:** the **OSS CAD Suite** from YosysHQ (GitHub: `YosysHQ/oss-cad-suite-build`) bundles Icarus, a recent Verilator, Yosys, SymbiYosys, GTKWave and more in one download for Linux and macOS. [Likely] **On Windows, use WSL2 (Ubuntu)**. Most of these tools are painful on native Windows. [Likely]

---

## 2. Step by step

### Step 1: Run the 4 × 4 systolic array test (already in this repo)

This exists and passes, so start by running it. [Certain]

```bash
pip install -r requirements-dev.txt
python3 -m pytest hw/testbenches/systolic_array -v
```

What is there:

- `hw/rtl/systolic_array/pe.sv`: one processing element (holds a weight, multiply-adds, passes data on).
- `hw/rtl/systolic_array/systolic_array.sv`: N × N weight-stationary array, INT8 in, INT32 out, with input skew and output deskew handled inside. Latency `2N − 1` cycles, one vector per cycle.
- `hw/testbenches/systolic_array/test_systolic_array.py`: three tests against numpy:
  1. identity weights,
  2. 50 random vectors back to back (including −128 edge cases),
  3. a **tiled dense layer** (8×16 @ 16×8 split into 4×4 tiles, then ReLU). This is the same pattern a real model layer uses.

I checked that the tests catch bugs: flipping `+` to `−` in the PE makes all three fail. [Certain]

**Exercises to learn the design:**

- Run a bigger array: `ARRAY_N=8 python3 -m pytest hw/testbenches/systolic_array`. I tested N = 2, 4 and 8, and all pass. [Certain]
- Dump a waveform (`waves=True` in `runner.test(...)`) and watch the diagonal wave of data in GTKWave.

### Step 2: Build the golden model in Python (no hardware yet)

Before more RTL, get a model running in **integer-only numpy** that matches a real framework. This is your reference for every later step.

1. Train a tiny **MNIST MLP**: 784 → 64 → 32 → 10, dense layers + ReLU only. It is all matrix multiplies, so it fits your array with no new hardware. [Likely]
2. Quantise to INT8 with **symmetric quantisation (zero point = 0)**. TFLite's default uses non-zero activation zero points, which adds subtraction logic in hardware. Avoid it for now. [Likely]
3. Write `sw/tools/golden.py` that runs the model using only integers: INT8 × INT8 → INT32 accumulate, then **requantise** (multiply by a fixed-point scale, shift, round, clamp to INT8), then ReLU.
4. Check accuracy matches the float model within ~1%. Save the weights and test inputs as `.npy` files.

**Exit check:** integer numpy model is within ~1% of float accuracy on MNIST. [Guessing on the exact tolerance]

### Step 3: Run one real layer through the RTL

Extend the cocotb test: load the trained weights from step 2, run layer 1 through the array with Python doing the tiling (like `test_tiled_dense_layer`), and compare the INT32 outputs to the golden model **bit for bit**.

**Exit check:** zero mismatches on 100 test images.

### Step 4: Run the whole model, still with Python as the "controller"

Run all three layers. Python does tiling, requantisation and ReLU between layers. The RTL only does matrix maths. Count cycles and compare to the theoretical minimum. This is your first performance number.

**Exit check:** same predicted digit as the golden model for every test image, plus a cycle count.

### Step 5: Move the controller into hardware, piece by piece

Now replace the Python pieces with RTL, one at a time, testing each against the golden model:

1. **Requantise + ReLU unit** after the array (the start of the vector unit).
2. **On-chip buffers** (SRAM models) for weights, activations and accumulators.
3. **Instruction decoder + sequencer** reading the ISA from `architecture-study.md` §6.5. Define the encoding in `bridge/isa/`.
4. **Assembler** in `bridge/assembler/` that turns the model into an instruction stream.
5. Run **Yosys** synthesis to check everything is synthesisable and see the size.

**Exit check:** the testbench only loads memory and an instruction stream, presses "go", and checks the final answer.

### Step 6: Bigger TinyML models

Switch to Verilator (≥ 5.036) for speed. Then work through models from the **MLPerf Tiny** benchmark suite, which are standard TinyML references: [Likely]

| Order | Model | Why this order |
|---|---|---|
| 1 | **Anomaly detection** (fully connected autoencoder) | Dense layers only, so no new hardware |
| 2 | **Keyword spotting** (DS-CNN) | First convolutions. Map them to matrix multiplies with "im2col" in software. Depthwise convs are a poor fit for a systolic array, so expect low utilisation and measure it. |
| 3 | **Image classification** (ResNet-8, CIFAR-10) | Adds skip connections (an element-wise add in the vector unit) |
| 4 | **Visual wake words** (MobileNetV1 0.25×) | Biggest of the four; a good stress test |

**Exit check per model:** bit-exact with the golden model, accuracy within 1% of the float model, cycles per inference recorded.

### Step 7: FPGA

Only after step 6. Synthesise with Vivado, add an AXI interface to the Zynq ARM cores, and run the README's 2 × 2 "hello world", then the MNIST MLP on real hardware. After that, the INT4 language model work from the main design doc.

---

## 3. Common traps

- **Signed maths in Verilog.** If one operand is unsigned, the whole expression becomes unsigned. Declare everything `signed` explicitly. The −128 test case in step 1 exists to catch this. [Certain]
- **Driving signals in cocotb's read-only phase.** After `await ReadOnly()` you must wait for a clock edge before driving inputs. The step 1 testbench hit this, and the fix is in `run_vectors`. [Certain]
- **Icarus accepts code Verilator rejects.** Run `verilator --lint-only -Wall` on every RTL change, even before you simulate with Verilator. [Certain]
- **Rounding mismatches** between your requantise RTL and the golden model. Pick one rounding rule (round half away from zero, or TFLite's), write it down in `bridge/isa/`, and use it in both.
- **Testing with all-random data only.** Also test zeros, max values, −128, and a matrix whose size is not a multiple of the array size.
