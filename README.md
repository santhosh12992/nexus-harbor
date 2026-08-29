# Harbor + Antigravity Custom-Skill Evaluation POC

A minimal, fully reproducible Proof of Concept (POC) demonstrating how to evaluate the **Google Antigravity SDK agent** with and without a custom skill using the **Harbor evaluation framework** in an isolated Docker sandbox.

---

## 1. Architecture Overview

```
Harbor Evaluation Framework (0.21.0+)
│
├── Docker Sandbox (Container Isolation)
│   ├── Base Environment: python:3.12-slim
│   ├── Workspace: /workspace/input.txt
│   └── Agent Runtime: /installed-agent/run_agent.py (uv + google-antigravity)
│
├── Benchmark Task (datasets/antigravity-skill-poc/task-001/)
│   ├── task.toml (Harbor Schema v1.4)
│   ├── instruction.md (summarize /workspace/input.txt)
│   └── environment/Dockerfile (isolated environment)
│
├── Custom Skill (skills/distinctive-output/)
│   └── SKILL.md (prepends "SKILL-MARKER: ENABLED" to report files)
│
└── Verifier (tests/test.sh)
    └── Checks /workspace/report.txt correctness and writes reward to /logs/verifier/reward.txt
```

---

## 2. Directory Structure

```
nexus-harbor/
├── README.md
├── pyproject.toml
├── requirements.txt
├── skills/
│   └── distinctive-output/
│       └── SKILL.md                 # Custom agent skill
├── datasets/
│   └── antigravity-skill-poc/
│       └── task-001/
│           ├── task.toml            # Harbor task definition (v1.4)
│           ├── instruction.md       # Task instruction for agent
│           ├── environment/
│           │   ├── Dockerfile       # Minimal task container
│           │   └── input.txt        # Input document to summarize
│           ├── tests/
│           │   └── test.sh          # Deterministic task verifier
│           └── solution/
│               └── solve.sh         # Reference solution (Oracle)
├── configs/
│   ├── antigravity-with-skill.yaml   # Harbor JobConfig with skill
│   └── antigravity-without-skill.yaml# Harbor JobConfig baseline
└── scripts/
    ├── run-with-skill.sh            # Run evaluation with skill
    ├── run-without-skill.sh         # Run evaluation without skill
    └── verify-poc.sh                # End-to-end POC verification script
```

---

## 3. How It Works

1. **Harbor's Native `antigravity-sdk` Agent**:
   Harbor provides pre-integrated support for Google Antigravity (`AgentName.ANTIGRAVITY_SDK`). During sandbox initialization, Harbor provisions `uv` and uploads `run_agent.py` to `/installed-agent/`.
2. **Skill Injection**:
   When `--skill` or `skills` in `JobConfig` is specified, Harbor copies the skill folder into `/harbor/skills` inside the sandbox and passes the path to `LocalAgentConfig(skills_paths=[...])` in the Antigravity SDK runtime.
3. **A/B Behavioral Difference**:
   - **Without Skill (Baseline)**: The agent follows standard summarization instructions and produces `/workspace/report.txt` containing the summary text.
   - **With Skill**: The agent discovers and reads `distinctive-output/SKILL.md`, prepending `SKILL-MARKER: ENABLED` as the very first line of `/workspace/report.txt`.
4. **Anti-Cheating Design**:
   The task instruction, Dockerfile, and environment do **not** mention or reveal the skill marker. The marker exists solely in `skills/distinctive-output/SKILL.md`.

---

## 4. Prerequisites

- **Python**: Python 3.12 or higher.
- **Docker**: Docker Desktop or Docker Engine running locally.
- **Harbor Framework**: Version 0.21.0 or higher (`pip install -r requirements.txt`).
- **Gemini API Key**: Set `GEMINI_API_KEY` for live Antigravity execution:
  ```bash
  export GEMINI_API_KEY="your-gemini-api-key"
  ```

---

## 5. Installation

```bash
# 1. Clone repository and navigate to workspace
git clone https://github.com/nexus/nexus-harbor.git
cd nexus-harbor

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. Running Evaluations

### Run Baseline (Without Skill)

```bash
./scripts/run-without-skill.sh
```
*Equivalent Harbor CLI command:*
```bash
harbor run -p datasets/antigravity-skill-poc/task-001 -a antigravity-sdk -m gemini-2.5-flash
```

### Run With Custom Skill

```bash
./scripts/run-with-skill.sh
```
*Equivalent Harbor CLI command:*
```bash
harbor run -p datasets/antigravity-skill-poc/task-001 -a antigravity-sdk -m gemini-2.5-flash --skill skills/distinctive-output
```

---

## 7. Expected Results

### Baseline Run (Without Skill)
`/workspace/report.txt`:
```text
The Pacific Ocean is Earth's largest ocean according to the provided text.
```
- **Verifier Reward**: `1.0` (Base task solved correctly)
- **First Line**: Summary text (No skill marker)

### Skill-Injected Run (With Skill)
`/workspace/report.txt`:
```text
SKILL-MARKER: ENABLED
The Pacific Ocean is Earth's largest ocean according to the provided text.
```
- **Verifier Reward**: `1.0` (Base task solved correctly)
- **First Line**: `SKILL-MARKER: ENABLED`

---

## 8. Verifying the POC Setup

Run the automated verification script:

```bash
./scripts/verify-poc.sh
```

This script verifies:
1. Harbor CLI and Python installation.
2. Docker availability.
3. Task and skill directory structure.
4. Schema validation against Harbor `TaskConfig` and `JobConfig` models.
5. Reference solution (Oracle) execution and reward generation (`1.0`).
6. A/B Skill marker detection logic.
7. Live agent execution (if Docker and `GEMINI_API_KEY` are present).

---

## 9. Why This Proves the POC

- **Control**: The task instruction, model (`gemini-2.5-flash`), Docker sandbox, and verifier are 100% identical between both runs.
- **Variable**: The only difference is the injection of `skills/distinctive-output/` via Harbor's skill mechanism.
- **Result**: The agent reads the injected skill and alters its output format to include `SKILL-MARKER: ENABLED` while preserving task accuracy.

---

## ⚡ TPU Kernel Speedup & Hardware Profiling Benchmark (MaxKernel on TPU v6e)

This benchmark evaluates AI coding agents on physical Google Cloud TPU hardware (v6e) optimizing JAX/Pallas kernels using real silicon XProf hardware traces.

### 1. Prerequisites & Environment Setup

1. **Google Cloud TPU Access**:
   Ensure your GCP project and Cloud TPU v6e instance are accessible:
   ```bash
   gcloud config set project tpu-prod-env-multipod
   gcloud compute tpus tpu-vm describe maxkernel-v6e-1 \
     --zone=asia-northeast1-b --project=tpu-prod-env-multipod
   ```

2. **Environment Configuration**:
   Create and populate `.env` from `.env.template`:
   ```bash
   cp .env.template .env
   ```
   Set your API keys and hardware targets in `.env`:
   ```bash
   GEMINI_API_KEY="your-gemini-api-key"
   TPU_NAME="maxkernel-v6e-1"
   TPU_ZONE="asia-northeast1-b"
   TPU_PROJECT="tpu-prod-env-multipod"
   USE_REAL_TPU="1"
   ```

3. **Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

---

### 2. TPU Verification Session Commands

You can run the TPU benchmark and verifier in multiple ways depending on your workflow:

#### A. Full Harbor Evaluation (Docker Sandbox + Physical TPU Hardware)
Runs the end-to-end benchmark inside Harbor's isolated container while executing kernels remotely on the physical TPU v6e:
```bash
./scripts/run-tpu-eval.sh
```
*Equivalent Harbor CLI command:*
```bash
harbor run -c configs/maxkernel-tpu-eval.yaml
```

#### B. MaxKernel Multi-Iteration Live Hardware Session
Runs the autonomous multi-turn optimization loop directly against the TPU v6e instance, pulling live XProf traces and computing speedups across iterations:
```bash
./scripts/run-tpu-session.sh session
```
*Direct Python invocation:*
```bash
USE_REAL_TPU=1 \
WORKSPACE_DIR=datasets/tpu-kernel-speedup/task-001/environment \
LOGS_DIR=/tmp/verifier_logs \
python3 datasets/tpu-kernel-speedup/task-001/tests/maxkernel_live_session.py
```

#### C. Standalone Gemini 2.5 Live Agent Loop
Runs an interactive agent using Gemini 2.5 + MaxKernel skill + physical TPU feedback loop:
```bash
./scripts/run-tpu-session.sh agent
```
*Direct Python invocation:*
```bash
USE_REAL_TPU=1 \
GEMINI_API_KEY="your-gemini-api-key" \
python3 datasets/tpu-kernel-speedup/task-001/tests/gemini_maxkernel_live_agent.py
```

#### D. Standalone Hardware Verifier
Executes the reference solution on physical hardware, validates numerical parity with the baseline, and records final verifier rewards:
```bash
./scripts/run-tpu-session.sh test
```
*Direct Bash invocation:*
```bash
USE_REAL_TPU=1 \
WORKSPACE_DIR=datasets/tpu-kernel-speedup/task-001/solution \
LOGS_DIR=/tmp/verifier_logs \
bash datasets/tpu-kernel-speedup/task-001/tests/test.sh
```

#### E. Local CPU Simulation Mode (No Hardware Needed)
Runs the test verifier using mock JAX execution without contacting Cloud TPU:
```bash
./scripts/run-tpu-session.sh sim
```
*Direct Python invocation:*
```bash
USE_REAL_TPU=0 \
WORKSPACE_DIR=datasets/tpu-kernel-speedup/task-001/solution \
LOGS_DIR=/tmp/verifier_logs \
python3 datasets/tpu-kernel-speedup/task-001/tests/profiler_runner.py
```

---

### 3. Expected Verification Results Matrix

| Iteration | Description | Target / Measured Latency | Speedup | Parity Status | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Reference Un-fused Attention Kernel | `350.00 μs` | `1.000x` | Baseline | `REFERENCE` |
| **Iter 1** | Block Tiling ($B_s = 64$) | `274.50 μs` | `1.275x` | `PASS` ($\Delta < 10^{-5}$) | `ACCEPTED` |
| **Iter 2** | Fused Contraction + Scaling ($B_s = 128$) | **`192.80 μs`** | **`1.815x`** | `PASS` ($\Delta < 10^{-5}$) | **`BEST (ACCEPTED)`** |
| **Iter 3** | Alternative Memory Tiling | `205.10 μs` | `1.706x` | `PASS` ($\Delta < 10^{-5}$) | `REJECTED (Regressed)` |

- **Target Speedup Threshold**: `≥ 1.20x`
- **Harbor Verifier Final Reward**: `1.0`

---

### 4. Viewing Interactive Trajectories

Launch Harbor's visual dashboard to inspect the agent's thoughts, tool calls, and TPU speedup progression:

```bash
harbor view jobs --host 0.0.0.0 --port 8085
```
Open `http://localhost:8085` in your browser.

