# This Dockerfile is here to help with end-to-end testing
# From flytekit
# $ docker build -f Dockerfile.dev --build-arg PYTHON_VERSION=3.10 -t localhost:30000/flytekittest:someversion .
# $ docker push localhost:30000/flytekittest:someversion
# From your test user code
# $ pyflyte run --image localhost:30000/flytekittest:someversion

ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-slim-bookworm

MAINTAINER Flyte Team <users@flyte.org>
LABEL org.opencontainers.image.source https://github.com/flyteorg/flytekit

WORKDIR /root

ARG VERSION
ARG TARGETARCH

COPY . /flytekit

# Note: Pod tasks should be exposed in the default image
# Note: Some packages will create config files under /home by default, so we need to make sure it's writable
# Note: There are use cases that require reading and writing files under /tmp, so we need to change its permissions.

# Run a series of commands to set up the environment:
# 1. Update and install dependencies.
# 2. Install Flytekit and its plugins.
# 3. Clean up the apt cache to reduce image size. Reference: https://gist.github.com/marvell/7c812736565928e602c4
# 4. Create a non-root user 'flytekit' and set appropriate permissions for directories.
RUN apt-get update && apt-get install build-essential vim libmagic1 git -y \
    && apt-get install wget -y \
    && mkdir -p /tmp/ \
    && mkdir -p /tmp/code-server \
    && wget --no-check-certificate -O /tmp/code-server/code-server-4.19.0-linux-${TARGETARCH}.tar.gz https://github.com/coder/code-server/releases/download/v4.19.0/code-server-4.19.0-linux-${TARGETARCH}.tar.gz \
    && tar -xzf /tmp/code-server/code-server-4.19.0-linux-${TARGETARCH}.tar.gz -C /tmp/code-server/ \
    && wget --no-check-certificate https://open-vsx.org/api/ms-python/python/2023.20.0/file/ms-python.python-2023.20.0.vsix -P /tmp/code-server \
    && wget --no-check-certificate https://open-vsx.org/api/ms-toolsai/jupyter/2023.9.100/file/ms-toolsai.jupyter-2023.9.100.vsix -P /tmp/code-server \
    && pip install --no-cache-dir -U --pre \
        flyteidl \
        -e /flytekit \
        -e /flytekit/plugins/flytekit-k8s-pod \
        -e /flytekit/plugins/flytekit-deck-standard \
        -e /flytekit/plugins/flytekit-flyteinteractive \
        scikit-learn \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && apt-get clean autoclean \
    && apt-get autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/ \
    && useradd -u 1000 flytekit \
    && chown flytekit: /root \
    && chown flytekit: /home \
    && :

ENV PYTHONPATH "/flytekit:/flytekit/plugins/flytekit-k8s-pod:/flytekit/plugins/flytekit-deck-standard:"

# Switch to the 'flytekit' user for better security.
USER flytekit

ENV PATH="/tmp/code-server/code-server-4.19.0-linux-${TARGETARCH}/bin:${PATH}"


# Install extensions using code-server
# Execution is performed here as code-server configuration depends on the USER setting
# If we install it as ROOT, the config will be stored in /root/.config/code-server/config.yaml
# Now, the config of code-server will be stored in /home/flytekit/.config/code-server/config.yaml
RUN code-server --install-extension /tmp/code-server/ms-python.python-2023.20.0.vsix \
    && code-server --install-extension /tmp/code-server/ms-toolsai.jupyter-2023.9.100.vsix
