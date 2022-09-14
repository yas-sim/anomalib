from openvino/ubuntu20_dev_no_samples:2022.1.0

USER root

RUN apt update && apt install -y sudo git curl vim xz-utils
RUN python3 -m pip install -U pip setuptools

RUN /opt/intel/openvino_2022/extras/scripts/download_opencv.sh
RUN source /opt/intel/openvino_2022/setupvars.sh

WORKDIR /root
RUN git clone https://github.com/openvinotoolkit/anomalib
WORKDIR /root/anomalib
RUN git checkout baca44920214d7a19a9f3bdebeb035a303ec1bcb

RUN python3 -m pip install anomalib

RUN mkdir -p /root/anomalib/datasets/MVTec && \
    curl -SL https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz | \
    tar -xJC /root/anomalib/datasets/MVTec

RUN sed -i -e "s/  export_mode: null/  export_mode: \"openvino\"/" anomalib/models/padim/config.yaml
RUN mkdir -p demos/padim
COPY anomalib_demo.py       demos/
COPY anomalib_demo_torch.py demos/
COPY background.png         demos/
COPY background_pyt.png     demos/
COPY padim/config.yaml      demos/padim/

ENV DISPLAY=:0

RUN df /run/shm -h
RUN python3 tools/train.py

#CMD ["python3", "demos/anomalib_demo.py"]

CMD ["/bin/bash"]
# docker build . --shm-size=256m --tag anomalib
# docker run -it -u 0 --shm-size=256m -e DISPLAY=192.168.1.51:0 --name anomalib anomalib:latest
