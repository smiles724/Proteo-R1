FROM verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.0-fa2.7.4

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN pip uninstall verl -y || true

RUN git clone --recurse-submodules https://github.com/rllm-org/rllm.git rllm

RUN cd rllm && \
    pip install -e ./verl && \
    pip install -e .

RUN pip install playwright && \
    playwright install chromium && \
    playwright install-deps

CMD ["/bin/bash"]

# Docker Usage
# docker build -t rllm .
# docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/rllm -v /tmp:/tmp --name rllm-container rllm sleep infinity
# docker start rllm-container
# docker exec -it rllm-container bash
