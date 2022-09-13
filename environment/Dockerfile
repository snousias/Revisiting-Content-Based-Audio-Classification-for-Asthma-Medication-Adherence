# hash:sha256:b24c64c1a9ab2932b9d1fbd705b1aeecf741da693359f4d9265f0d6fd985f5c9
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        keras-gpu==2.4.3 \
        matplotlib==3.5.1 \
        pandas==1.4.2 \
        scikit-learn==1.0.2 \
        tensorflow==2.4.1 \
    && conda clean -ya
