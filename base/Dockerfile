FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

USER root
RUN apt-get update \
        && apt-get install -y --no-install-recommends \
               curl \
               locales \
               vim-tiny \
               wget \
               build-essential \
               apt-transport-https \
               software-properties-common \
               unzip \
               libpq-dev \
               libreadline-dev \
               libbz2-dev \
               lib32ncurses5-dev \
               libreadline6-dev \
               liblzma-dev \
               libbz2-dev \
               libpcre3-dev \
               libffi-dev \
               language-pack-ko \
               libcurl4-openssl-dev \
               libssl-dev \
               procps \
               ca-certificates \
               git

#RUN if [ ! -d "/home/splunk" ]; then \
#    adduser --disabled-password --gecos "" splunk -u 2001\
#    && addgroup splunk staff ; fi

RUN adduser --disabled-password --gecos "" splunk -u 2001\
    && addgroup splunk staff

USER splunk
WORKDIR /home/splunk

#RUN if [ ! -d "/home/splunk" ]; then \
#    curl -fsSLo Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
#    && bash Miniforge3.sh -p /home/splunk/miniforge3 -b \
#    && rm Miniforge3.sh ; fi

#ENV PATH /home/splunk/miniforge3/bin:${PATH}
#ENV bash /bin/bash

#RUN if [ ! -d "/home/splunk" ]; then \
#    /home/splunk/miniforge3/bin/conda install python=3.6.12 -y \
#    && /home/splunk/miniforge3/bin/pip install -U pip \
#    && /home/splunk/miniforge3/bin/pip install --no-cache-dir pyyaml requests sendslack pandas sklearn sh boto3 ; fi

RUN curl -fsSLo Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
RUN bash Miniforge3.sh -p /home/splunk/miniforge3 -b
RUN rm Miniforge3.sh
ENV PATH /home/splunk/miniforge3/bin:${PATH}
ENV bash /bin/bash

RUN /home/splunk/miniforge3/bin/conda install python=3.6.12 -y
RUN /home/splunk/miniforge3/bin/pip install -U pip
RUN /home/splunk/miniforge3/bin/pip install --no-cache-dir pyyaml requests sendslack pandas sklearn sh boto3

COPY default.py /home/splunk/default.py

ARG GIT_BUILD_ID_ARG
ENV GIT_BUILD_ID $GIT_BUILD_ID_ARG
ENV PROMPT_COMMAND 'history -a'
ENV LANG 'ko_KR.utf8'

COPY run.py /home/splunk/run.py

ENTRYPOINT ["python3", "/home/splunk/run.py"]
