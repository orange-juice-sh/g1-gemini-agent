# Blackwell 世代の GPU でも動くよう CUDA 12.8 以降を使う
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ca-certificates unzip \
        libegl1 libgl1 libglvnd0 libosmesa6 libglfw3 \
        ffmpeg fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA の EGL を使う（ランタイムが用意しない場合の保険）
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Ubuntu 24.04 はシステムの pip に入れられないので venv を使う
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

RUN git clone --depth 1 https://github.com/unitreerobotics/unitree_rl_gym.git /opt/unitree_rl_gym

# マウントしたファイルが root の持ち物にならないよう、ホストと同じ UID/GID で動かす
ARG UID=1000
ARG GID=1000
RUN (userdel -r ubuntu 2>/dev/null || true) && \
    groupadd -o -g ${GID} dev && \
    useradd -o -m -u ${UID} -g ${GID} -s /bin/bash dev
USER dev

ENV UNITREE_RL_GYM=/opt/unitree_rl_gym \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace
CMD ["sleep", "infinity"]
