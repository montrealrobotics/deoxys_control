
ARG TARGET_ENV=cpu

FROM nvidia/cuda:12.5.0-devel-ubuntu22.04 AS env-gpu
FROM ubuntu:22.04 AS env-cpu

FROM env-${TARGET_ENV} AS final

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

ENV ROS_DISTRO=humble

#RUN sed -i '/kitware/d' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    curl gnupg2 lsb-release software-properties-common build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends -y \
    libusb-1.0-0-dev \
    libudev-dev \
    libgtk-3-dev \
    libglfw3-dev \
    python3-pip \
    git \
    xauth \
    vim \
    wget && apt-get clean

RUN mkdir -p /home/hw_ctrl_ws/src

WORKDIR /home/hw_ctrl_ws/src
COPY deoxys_control/deoxys/requirements_xarm.txt /home/requirements_xarm.txt
WORKDIR /home/
RUN pip install -r requirements_xarm.txt
ENV PYTHONPATH=/home/hw_ctrl_ws/src/deoxys_control/deoxys:${PYTHONPATH}

RUN pip3 install zmq xarm-python-sdk pyquaternion numpy pyRobotiqGripper
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
WORKDIR /home/hw_ctrl_ws/src/deoxys_control
CMD ["bash"]
