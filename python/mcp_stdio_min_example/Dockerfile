FROM ubuntu:24.04

RUN apt-get update 
RUN apt-get install -y \
    python3 \
    python3-pip

RUN pip install uv --break-system-packages

COPY mcp_server /mcp_server

WORKDIR /mcp_server

RUN uv sync

CMD ["uv", "run", "minexample"]