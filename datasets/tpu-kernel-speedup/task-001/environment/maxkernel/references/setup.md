# MaxKernel TPU VM & Local Setup Guide

This guide describes how to set up the Python virtual environment
(`maxkernel_venv`) and install dependencies on both your **local Cloudtop** and
the **TPU VM**.

--------------------------------------------------------------------------------

## Part 1: Local Cloudtop Setup

Perform these steps on your local Cloudtop.

### Step 1: Verify Python Version

Ensure you have Python 3.12 installed:

```bash
python3 --version
```

If you need to install it:

```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
```

### Step 2: Verify if Virtual Environment Exists

Run the following command to check if `maxkernel_venv` already exists and is
valid:

```bash
if [ -x ~/maxkernel_venv/bin/python3 ]; then echo "✅ maxkernel_venv exists"; else echo "❌ maxkernel_venv NOT found"; fi
```

If it does not exist, create it:

```bash
python3.12 -m venv ~/maxkernel_venv
```

### Step 3: Install Dependencies

Activate the virtual environment and install the required packages:

```bash
# Activate the venv
source ~/maxkernel_venv/bin/activate

# Upgrade pip
pip install --upgrade pip --index-url https://pypi.org/simple

# Install dependencies from the combined requirements.txt
pip install -r maxkernel/references/requirements.txt --index-url https://pypi.org/simple
```

### Step 4: Verification

Verify the installation:

```bash
# 1. Verify we are using the venv python
which python3
# Expected: /usr/local/google/home/<username>/maxkernel_venv/bin/python3 (or similar path under your home)

# 2. Verify JAX version is 0.11.0
python3 -c "import jax; print(jax.__version__)"
# Expected: 0.11.0
```

--------------------------------------------------------------------------------

## Part 2: TPU Configuration (Local vs Remote VM)

You can run the agent either on a **Local VM / Cloudtop** (accessing a remote TPU VM) or **directly on the TPU VM** (local execution).

> **Note:** When this package is installed via Nexus, connection settings are
> normally supplied by the install-time questionnaire and read from
> `.coworker/torchtpu-agents/environment.json`, which takes priority. The
> `tpu_config.json` file below is the fallback used when no Coworker
> environment is present.

### Step 1: Create TPU Configuration

Create a `tpu_config.json` file in `maxkernel/`:

#### Option A: Remote TPU VM Mode (Agent on Cloudtop accessing remote TPU VM)
```json
{
    "mode": "remote",
    "tpu_name": "<YOUR_TPU_NAME>",
    "zone": "<YOUR_TPU_ZONE>",
    "project": "<YOUR_GCP_PROJECT>"
}
```
*(If `"mode"` is omitted but `tpu_name` is present, it defaults to `remote` mode for backward compatibility).*

#### Option B: Local TPU VM Mode (Agent running on the TPU VM directly)
```json
{
    "mode": "local"
}
```

### Step 2: Running with `tpu_client.py`

The `tpu_client.py` will automatically read `tpu_config.json` and start the server daemon (locally or via remote SSH/SCP depending on mode). You can also override the target mode using the `--mode` CLI flag (`--mode local` or `--mode remote`).

--------------------------------------------------------------------------------

## Part 3: Async Job Queue & Client Usage

The TPU Server features an asynchronous job queue for handling execution requests without conflicts when the TPU is busy.

### Standard Synchronous Call with Queue Waiting
Submits job to queue and automatically polls until completion:
```bash
python3 maxkernel/scripts/tpu_client.py --action correctness_test --code_file path/to/script.py
```

### Non-blocking Async Submission
Submits job and returns immediately with a `job_id`:
```bash
python3 maxkernel/scripts/tpu_client.py --action autotune --code_file payload.json --submit_only
```

### Checking Job Status & Results
```bash
python3 maxkernel/scripts/tpu_client.py --check_job job_1700000000000_abc123
```

### Inspecting TPU Server Queue
```bash
python3 maxkernel/scripts/tpu_client.py --queue
```
