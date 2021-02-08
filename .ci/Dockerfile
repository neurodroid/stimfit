FROM ubuntu
MAINTAINER Dilawar Singh <dilawar.s.rajput@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update

RUN apt install -y cmake git python3-dev python3-numpy python3-setuptools \
    python3-wheel libhdf5-dev libjpeg-dev swig python3-wxgtk4.0 \
    libwxgtk3.0-gtk3-dev wx-common fftw3-dev liblapack-dev \
    cmake python3-sip-dev

COPY . /root

WORKDIR /root

RUN mkdir _build && cd _build && cmake .. && make -j$(nproc) && make install
RUN python3 -c "import stfio"
