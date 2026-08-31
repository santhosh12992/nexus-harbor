FROM python:3.11-slim
WORKDIR /workspace
RUN pip install --no-cache-dir jax jaxlib numpy && mkdir -p /logs/verifier /logs/agent /logs/artifacts /workspace /tests
COPY baseline_kernel.py /workspace/baseline_kernel.py
COPY profile_kernel.py /workspace/profile_kernel.py
RUN chmod +x /workspace/profile_kernel.py
CMD ["/bin/bash"]
