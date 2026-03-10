FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set noninteractive mode to avoid prompts during installation
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

RUN pip3 install zmq xarm-python-sdk pyquaternion numpy pyRobotiqGripper
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
WORKDIR /home/hw_ctrl_ws/src
CMD ["bash"]
