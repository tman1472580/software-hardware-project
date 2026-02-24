# Edge AI Hardware/Software Co-Design

> **An end-to-end hardware/software co-design project building a custom TPU-style FPGA accelerator and a bare-metal inference engine for quantized Small Language Models (SLMs).**

From Verilog to LLMs: Bridging silicon and software by designing a custom systolic array on an FPGA and writing the low-level C++/Rust inference kernels to run AI at the edge. 

This repository features a full-stack AI edge deployment featuring a custom ISA hardware accelerator (FPGA) and a highly optimized INT4 inference engine with custom GEMM and attention kernels.

---

## 📂 Project Folder Structure

By defining a custom Instruction Set Architecture (ISA) and building both the silicon logic and the software stack from scratch, this project is split into distinct tracks that meet in the middle (`/bridge`):

```text
├── hw/                     # 🖥️ HARDWARE TRACK (FPGA & RTL)
│   ├── rtl/                # SystemVerilog/Verilog source code
│   │   ├── systolic_array/ # The matrix multiplication engine
│   │   ├── memory/         # SRAM controllers and memory fetch logic
│   │   └── control/        # Instruction decoder and control unit
│   ├── testbenches/        # Hardware simulation files to verify RTL logic
│   └── fpga/               # Board-specific files (e.g., Xilinx constraint files, synthesis scripts)
│
├── sw/                     # ⚙️ SOFTWARE TRACK (Inference Engine)
│   ├── engine/             # Main C++/Rust inference loop, KV Cache management
│   ├── kernels/            # Custom low-level implementations (GEMM, FlashAttention)
│   ├── drivers/            # Host-to-FPGA communication (PCIe/USB data transfer)
│   └── tools/              # Python scripts to quantize PyTorch/HF models to custom INT4
│
├── bridge/                 # 🤝 THE MEETING POINT
│   ├── isa/                # Shared Instruction Set definitions (opcodes, C++ headers)
│   ├── assembler/          # Script to convert custom assembly into machine code
│   └── simulator/          # Cycle-accurate C++/Python simulator of the hardware
│
├── tests/                  # End-to-end integration tests (Model -> SW -> Simulator/HW)
└── docs/                   # Architecture diagrams and the official ISA manual

Suggested project structure:


HARDWARE TRACK (FPGA)                                SOFTWARE TRACK (Inference Engine)
             |                                                          |
             v                                                          v
=================================================================================================
|                           STAGE 1: THE CONTRACT (Defining the rules)                          |
|  Both sides agree on:                                                                         |
|  1. The ISA (What assembly instructions will the hardware understand?)                        |
|  2. Data Format (e.g., INT4 for weights, INT32 for accumulation)                              |
|  3. Memory Map (How will the CPU talk to the FPGA?)                                           |
=================================================================================================
             |                                                          |
             v                                                          v
+-----------------------------+                          +--------------------------------------+
| STAGE 2: HW PROTOTYPING     |                          | STAGE 2: SW PROTOTYPING              |
| - Write RTL/Verilog         |                          | - Write basic engine (C++/Rust)      |
| - Design Systolic Array     |                          | - Build the Tokenizer                |
| - Quantize the SLM weights to INT4                     |
+-----------------------------+                          +--------------------------------------+
             |                                                          |
             v                                                          v
+-----------------------------+                          +--------------------------------------+
| STAGE 3: THE SIMULATOR      |                          | STAGE 3: KERNEL DEVELOPMENT          |
| - Compile the Verilog into  | <--- (SW tests on HW)--- | - Write custom GEMM/Attention kernels|
|   a cycle-accurate software |                          | - Compile kernels using the new ISA  |
|   simulator.                |                          | - Run them on the HW simulator       |
+-----------------------------+                          +--------------------------------------+
             |                                                          |
             v                                                          v
=================================================================================================
|                             STAGE 4: SILICON BRING-UP (The physical merge)                    |
|  - Flash the actual FPGA board with the hardware design.                                      |
|  - Software engineer writes the Host-to-Device drivers (PCIe, USB, etc.).                     |
|  - The "Hello World" test: Send a single 2x2 matrix multiplication and verify the answer.     |
=================================================================================================
                                              |
                                              v
=================================================================================================
|                             STAGE 5: FULL INTEGRATION & PROFILING                             |
|  - Feed the fully quantized SLM into the inference engine.                                    |
|  - Run the Prefill phase (heavy matrix math) and Decode phase (token-by-token generation).    |
|  - Profile the system:                                                                        |
|      * Is the hardware sitting idle? -> Software needs to optimize memory fetching.           |
|      * Is the math too slow? -> Hardware needs to optimize the systolic array.                |
=================================================================================================
