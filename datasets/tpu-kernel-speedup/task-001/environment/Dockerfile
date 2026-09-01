FROM python:3.11-slim
WORKDIR /workspace
RUN pip install --no-cache-dir jax jaxlib numpy fastapi uvicorn pydantic && mkdir -p /logs/verifier /logs/agent /logs/artifacts /workspace /tests /workspace/.agent/agents /workspace/maxkernel /workspace/.coworker
COPY baseline_kernel.py /workspace/baseline_kernel.py
COPY base.py /workspace/base.py
COPY profile_kernel.py /workspace/profile_kernel.py
COPY state.json /workspace/state.json
COPY maxkernel/ /workspace/maxkernel/
COPY .agent/ /workspace/.agent/
COPY .coworker/ /workspace/.coworker/
RUN chmod +x /workspace/profile_kernel.py
CMD ["/bin/bash"]
