# Build from the repo root: docker build -f dagshare_exp/deploy/Dockerfile .
FROM vllm/vllm-openai:v0.10.2

ARG GIT_SHA=unknown
ARG SOURCE_URL=https://github.com/azkanab/vllm-continuum
LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.description="vLLM v0.10.2 + Continuum + DAGShare (Python overlay)"

COPY vllm /tmp/fork-vllm
RUN set -eux; \
    VLLM_DIR="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"; \
    echo "overlaying fork onto ${VLLM_DIR}"; \
    cd /tmp/fork-vllm; \
    find . -type f ! -name '*.so' ! -name '*.pyc' ! -path '*/__pycache__/*' ! -name '_version.py' \
        -exec install -D -m 0644 '{}' "${VLLM_DIR}/{}" ';'; \
    rm -rf /tmp/fork-vllm; \
    python3 -m compileall -q "${VLLM_DIR}/v1/core" "${VLLM_DIR}/entrypoints"; \
    python3 -c 'import vllm.v1.core.sched.dagshare, vllm.v1.core.sched.request_queue'

COPY dagshare_exp /workspace/engine/dagshare_exp
RUN set -eux; \
    cd /workspace/engine/dagshare_exp; \
    reqs="-r deploy/requirements-image.txt"; \
    if [ -f requirements.txt ]; then reqs="${reqs} -r requirements.txt"; fi; \
    if command -v uv >/dev/null 2>&1; then uv pip install --system --no-cache ${reqs}; \
    else python3 -m pip install --no-cache-dir ${reqs}; fi; \
    chmod +x deploy/entrypoint.sh; \
    python3 prepare_sharegpt.py --out data/sharegpt.jsonl --cache-dir /tmp/sharegpt; \
    rm -rf /tmp/sharegpt

ENV VLLM_DAGSHARE_GIT_SHA=${GIT_SHA} \
    PYTHONPATH=/workspace/engine \
    PYTHONUNBUFFERED=1
WORKDIR /workspace/engine
ENTRYPOINT ["/workspace/engine/dagshare_exp/deploy/entrypoint.sh"]
CMD ["--help"]
