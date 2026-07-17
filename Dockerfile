# LongHaul-Bench on-device runtime — resource-capped reproducible environment.
#
# Build:  docker build -t longhaul-bench .
# Run  :  docker run --memory=8g --cpus=4 -v ./local/models:/models longhaul-bench \
#             python agents/longrun.py --world runs/v01/world.json \
#             --episodes runs/v01/episodes.jsonl --operator reflect \
#             --policy compress --limit 1000 --out runs/docker_run
#
# The 8 GB / 4-CPU caps ARE the experiment: they enforce the edge budget
# identically on any host. Model files are mounted, not baked in.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp server (Linux x64 CPU build, pinned release)
ARG LLAMA_RELEASE=b9918
RUN curl -sL -o /tmp/llama.tgz \
    "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_RELEASE}/llama-${LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz" \
    && mkdir -p /opt/llama && tar -xzf /tmp/llama.tgz -C /opt/llama --strip-components=1 && rm /tmp/llama.tgz \
    && SERVER=$(find /opt/llama -name llama-server -type f | head -1) \
    && ln -s "$SERVER" /usr/local/bin/llama-server \
    && LIBDIR=$(dirname "$SERVER") && echo "$LIBDIR" > /etc/ld.so.conf.d/llama.conf && ldconfig

WORKDIR /app
RUN pip install --no-cache-dir qdrant-client psutil

COPY environments/ environments/
COPY agents/ agents/
COPY scripts/ scripts/
COPY runs/v01/ runs/v01/

# start the SLM server in the background, then run the given command
ENTRYPOINT ["/bin/sh", "-c", "llama-server -m /models/qwen2.5-3b-instruct-q4_k_m.gguf --port 8080 -c 4096 & sleep 20 && exec \"$@\"", "--"]
CMD ["python", "agents/longrun.py", "--help"]
