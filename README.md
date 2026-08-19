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
