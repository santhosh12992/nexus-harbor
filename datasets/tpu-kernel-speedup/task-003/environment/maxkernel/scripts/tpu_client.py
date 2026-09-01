# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

SERVER_URL = "http://127.0.0.1:8000"


def get_tpu_config():
    """Read configuration from Coworker environment or fallback to tpu_config.json."""
    # 1. Priority: Coworker environment.json
    for candidate in [
            os.path.expanduser(".coworker/torchtpu-agents/environment.json"),
            os.path.join(os.getcwd(), ".coworker", "torchtpu-agents",
                         "environment.json"),
    ]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r") as f:
                    env = json.load(f)
                    remote_host = env.get("remote_host", "").strip()
                    is_local = remote_host in ("", "localhost", "127.0.0.1",
                                               "local-tpu-vm")
                    return {
                        "mode": "local" if is_local else "remote",
                        "tpu_name": remote_host if not is_local else "",
                        "ssh_user": env.get("ssh_user", "user"),
                        "zone": env.get("zone", "us-central2-b"),
                        "project": env.get("project", ""),
                    }
            except Exception as e:
                logging.warning(
                    f"Could not parse Coworker environment.json: {e}")

    # 2. Fallback: local tpu_config.json
    config_path = os.path.join(os.path.dirname(__file__), "..",
                               "tpu_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to read tpu_config.json: {e}")
    return None


def check_health():
    """Check if the TPU server is running and reachable."""
    req = urllib.request.Request(f"{SERVER_URL}/health")
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, ConnectionError):
        return False


def ssh_cmd(config, cmd, bg=False):
    """Run an SSH command on the TPU VM."""
    if config.get("zone") and config.get("project"):
        base = [
            "/usr/bin/gcloud", "compute", "tpus", "tpu-vm", "ssh",
            config["tpu_name"], "--zone", config["zone"], "--project",
            config["project"], "--command", cmd
        ]
    else:
        user_host = f"{config.get('ssh_user', 'user')}@{config['tpu_name']}" if config.get(
            "ssh_user") else config["tpu_name"]
        base = ["ssh", "-o", "StrictHostKeyChecking=no", user_host, cmd]
    if bg:
        return subprocess.Popen(base,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True)
    return subprocess.run(base, capture_output=True, text=True)


def scp_cmd(config, local_path, remote_path):
    """SCP a file to the TPU VM."""
    if config.get("zone") and config.get("project"):
        base = [
            "/usr/bin/gcloud", "compute", "tpus", "tpu-vm", "scp", local_path,
            f"{config['tpu_name']}:{remote_path}", "--zone", config["zone"],
            "--project", config["project"]
        ]
    else:
        user_host = f"{config.get('ssh_user', 'user')}@{config['tpu_name']}" if config.get(
            "ssh_user") else config["tpu_name"]
        base = [
            "scp", "-o", "StrictHostKeyChecking=no", local_path,
            f"{user_host}:{remote_path}"
        ]
    return subprocess.run(base, capture_output=True, text=True)


def start_server_idempotent(mode=None):
    """Lazily start the server if it is not running."""
    if check_health():
        return True

    config = get_tpu_config() or {}
    target_mode = mode or config.get("mode")
    if not target_mode:
        if "tpu_name" in config:
            target_mode = "remote"
        else:
            target_mode = "local"

    if target_mode not in ("local", "remote"):
        logging.error(
            f"Invalid mode '{target_mode}'. Mode must be 'local' or 'remote'.")
        return False

    LOCK_FILE = "/tmp/maxkernel_tpu_setup.lock"
    lock_fd = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not check_health():
            if target_mode == "local":
                logging.info(
                    "TPU Server is down. Initiating local lazy start...")
                server_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "tpu_server.py"))
                log_file_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..",
                                 "server.log"))
                logging.info(f"Booting local server daemon: {server_path}")
                with open(log_file_path, "a") as log_f:
                    subprocess.Popen(
                        [sys.executable, server_path],
                        stdout=log_f,
                        stderr=log_f,
                        start_new_session=True,
                    )
                for _ in range(15):
                    if check_health():
                        logging.info("Local TPU Server is up and connected!")
                        break
                    time.sleep(1)
                else:
                    logging.error(
                        "Local TPU Server failed to boot within timeout.")
                    return False
            else:
                if not all(k in config
                           for k in ("tpu_name", "zone", "project")):
                    logging.error(
                        "Remote TPU Configuration missing! `tpu_config.json` must contain tpu_name, zone, and project for remote mode."
                    )
                    return False

                logging.info(
                    "TPU Server is down. Initiating remote lazy start...")
                # 1. Sync necessary files to the remote TPU
                req_path = os.path.join(os.path.dirname(__file__), "..",
                                        "references", "requirements.txt")
                server_path = os.path.join(os.path.dirname(__file__),
                                           "tpu_server.py")

                # Ensure the target directory structure exists
                ssh_cmd(config, "mkdir -p ~/maxkernel_deploy")

                logging.info(
                    "Syncing requirements and server script to TPU VM...")
                if os.path.exists(req_path):
                    scp_cmd(config, req_path,
                            "~/maxkernel_deploy/requirements.txt")
                scp_cmd(config, server_path,
                        "~/maxkernel_deploy/tpu_server.py")

                # 2. Setup VENV and run server remotely
                setup_script = """
                cd ~/maxkernel_deploy
                if [ ! -d "$HOME/maxkernel_venv" ]; then
                    echo "Setting up venv..."
                    sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
                    python3.12 -m venv ~/maxkernel_venv
                    source ~/maxkernel_venv/bin/activate
                    python3 -m pip install --upgrade pip
                    if [ -f requirements.txt ]; then
                        python3 -m pip install -r requirements.txt --quiet
                    else
                        python3 -m pip install jax[tpu]==0.11.0 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
                    fi
                fi
                source ~/maxkernel_venv/bin/activate
                # Kill any stale python servers running on 8000
                fuser -k 8000/tcp || true
                # Start the server daemon detached
                nohup python3 tpu_server.py > server.log 2>&1 &
                """

                logging.info(
                    "Executing remote setup and booting server (this may take a few minutes)..."
                )
                res = ssh_cmd(config, setup_script)
                if res.returncode != 0:
                    logging.error(f"Remote setup failed: {res.stderr}")
                    return False

                # 3. Establish Local Port Forwarding Tunnel
                logging.info("Establishing SSH tunnel for port 8000...")
                tunnel_cmd = [
                    "/usr/bin/gcloud", "compute", "tpus", "tpu-vm", "ssh",
                    config["tpu_name"], "--zone", config["zone"], "--project",
                    config["project"], "--", "-N", "-L", "8000:localhost:8000"
                ]
                subprocess.Popen(tunnel_cmd,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)

                # Wait for tunnel and server to come up
                for _ in range(30):
                    if check_health():
                        logging.info(
                            "Remote Server is up and connected via tunnel!")
                        break
                    time.sleep(1)
                else:
                    logging.error(
                        "Server failed to boot or tunnel failed to connect within timeout."
                    )
                    return False

    except BlockingIOError:
        logging.info("Another setup is in progress. Waiting...")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if not check_health():
            logging.error("Setup lock released, but server is still down!")
            return False
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    return True


def strip_markdown(code: str) -> str:
    code_content = code.strip()
    if code_content.startswith("```"):
        lines = code_content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return code_content


def submit_job(action: str, code: str, timeout: int):
    endpoint = f"{SERVER_URL}/submit"
    if action == "autotune":
        try:
            payload_dict = json.loads(code)
            if "timeout" not in payload_dict:
                payload_dict["timeout"] = 120
            if "total_timeout" not in payload_dict:
                payload_dict["total_timeout"] = max(timeout, 1800)
            autotune_req = payload_dict
        except json.JSONDecodeError:
            autotune_req = {
                "code_template": code,
                "search_space": {},
                "timeout": 120,
                "total_timeout": max(timeout, 1800),
            }
        submission = {"action": action, "autotune_request": autotune_req}
    else:
        code_req = {"code": code, "timeout": max(timeout, 180)}
        submission = {"action": action, "code_request": code_req}

    payload = json.dumps(submission).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode()
        logging.error(f"HTTPError submitting job: {error_msg}")
        return None
    except Exception as e:
        logging.error(f"Error submitting job: {e}")
        return None


def check_job_status(job_id: str):
    endpoint = f"{SERVER_URL}/job/{job_id}"
    req = urllib.request.Request(endpoint, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        logging.error(f"Error checking status for job '{job_id}': {e}")
        return None


def get_queue_info():
    endpoint = f"{SERVER_URL}/queue"
    req = urllib.request.Request(endpoint, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        logging.error(f"Error checking queue: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="TPU Execution Client wrapper with Async Job Queue support."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "remote"],
        default=None,
        help=(
            "Target execution environment: 'local' (agent running on TPU VM) or"
            " 'remote' (agent accessing TPU VM remotely). Defaults to mode in"
            " tpu_config.json."),
    )
    parser.add_argument(
        "--action",
        choices=[
            "compilation_test",
            "correctness_test",
            "performance_test",
            "profile",
            "autotune",
        ],
        help="Endpoint to call on server.",
    )
    parser.add_argument("--code_file",
                        help="Path to Python code or JSON payload to execute.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for execution (default 600s).",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=2.0,
        help="Interval in seconds to poll job status when queued.",
    )
    parser.add_argument(
        "--submit_only",
        action="store_true",
        help="Submit job to queue and exit immediately with job_id.",
    )
    parser.add_argument("--check_job",
                        help="Check status of a specific job_id.")
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Show current queue status of TPU server.",
    )
    args = parser.parse_args()

    if not start_server_idempotent(mode=args.mode):
        logging.error("Cannot proceed: TPU server is unreachable.")
        sys.exit(1)

    if args.queue:
        q_info = get_queue_info()
        if q_info:
            print("\n" + "=" * 40)
            print("TPU Server Queue Status:")
            print("=" * 40)
            print(f"Total Jobs: {q_info.get('total_jobs')}")
            print(f"Queued Count: {q_info.get('queued_count')}")
            print(f"Running Count: {q_info.get('running_count')}")
            if q_info.get("running_jobs"):
                print(f"Running Jobs: {q_info['running_jobs']}")
            if q_info.get("queued_jobs"):
                print(f"Queued Jobs: {q_info['queued_jobs']}")
        else:
            print("Failed to retrieve queue info.")
        sys.exit(0)

    if args.check_job:
        job_info = check_job_status(args.check_job)
        if not job_info:
            logging.error(f"Job '{args.check_job}' not found.")
            sys.exit(1)

        status = job_info.get("status")
        pos = job_info.get("queue_position", 0)
        res = job_info.get("result") or {}

        print("\n" + "=" * 40)
        print("Job Status Report:")
        print("=" * 40)
        print(f"Job ID: {job_info.get('job_id')}")
        print(f"Action: {job_info.get('action')}")
        print(f"Status: {status}")
        if status == "queued":
            print(f"Queue Position: {pos}")
        if job_info.get("created_at"):
            print(f"Created At: {time.ctime(job_info['created_at'])}")
        if job_info.get("completed_at"):
            print(f"Completed At: {time.ctime(job_info['completed_at'])}")

        if res:
            print(f"Exit Code: {res.get('exit_code', 1)}")
            if res.get("error"):
                print(f"\nSTDERR:\n{res['error']}")
            if res.get("output"):
                print(f"\nSTDOUT:\n{res['output']}")
        sys.exit(0 if (
            status == "completed" and res.get("exit_code") == 0) else 1)

    if not args.action or not args.code_file:
        parser.error(
            "Both --action and --code_file are required unless using --check_job or"
            " --queue.")

    if not os.path.exists(args.code_file):
        logging.error(f"File not found: {args.code_file}")
        sys.exit(1)

    with open(args.code_file, "r") as f:
        code = strip_markdown(f.read())

    effective_timeout = (max(args.timeout, 1800)
                         if args.action == "autotune" else args.timeout)
    logging.info(f"Submitting '{args.action}' request to TPU job queue"
                 f" (timeout={effective_timeout}s)...")
    submit_res = submit_job(args.action, code, effective_timeout)

    if not submit_res or "job_id" not in submit_res:
        logging.error("Failed to submit job to TPU server.")
        sys.exit(1)

    job_id = submit_res["job_id"]
    status = submit_res.get("status", "queued")
    pos = submit_res.get("queue_position", 0)

    if args.submit_only:
        print("\n" + "=" * 40)
        print("Job Submitted (Async):")
        print("=" * 40)
        print(f"Job ID: {job_id}")
        print(f"Status: {status}")
        print(f"Queue Position: {pos}")
        print("AGENT INSTRUCTION: Your request is enqueued. Use '--check_job"
              f" {job_id}' to poll results later.")
        sys.exit(0)

    logging.info(
        f"Job '{job_id}' submitted. Status: {status} (Queue position: {pos})")
    if status == "queued":
        logging.info(
            "TPU is currently busy. Job is waiting in queue. Polling for"
            " completion...")

    start_time = time.time()
    last_status = None
    last_pos = None
    max_wait = (effective_timeout + 1200
                )  # allow ample margin for queue waiting + execution

    while True:
        job_info = check_job_status(job_id)
        if job_info:
            current_status = job_info.get("status")
            current_pos = job_info.get("queue_position", 0)

            if current_status != last_status or current_pos != last_pos:
                if current_status == "queued":
                    logging.info(
                        f"Job '{job_id}' is WAITING in queue (Position: {current_pos})..."
                    )
                elif current_status == "running":
                    logging.info(f"Job '{job_id}' is now RUNNING on TPU...")
                last_status = current_status
                last_pos = current_pos

            if current_status in ("completed", "failed", "cancelled"):
                res = job_info.get("result") or {}
                print("\n" + "=" * 40)
                print("Execution Result:")
                print("=" * 40)
                print(f"Job ID: {job_id}")
                print(f"Status: {current_status}")
                print(f"Exit Code: {res.get('exit_code', 1)}")
                if res.get("error"):
                    print(f"\nSTDERR:\n{res['error']}")
                if res.get("output"):
                    print(f"\nSTDOUT:\n{res['output']}")

                sys.exit(0 if (
                    current_status == "completed" and res.get("exit_code") == 0
                ) else 1)

        if time.time() - start_time > max_wait:
            logging.error(
                f"Client timed out after waiting for job '{job_id}' to complete in"
                " queue.")
            sys.exit(1)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
