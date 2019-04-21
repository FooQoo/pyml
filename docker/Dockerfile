FROM nvidia/cuda:8.0-cudnn6-runtime
ENV PYTHONENCODING "utf-8"

RUN apt-get update
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y build-essential language-pack-ja
RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-setuptools
RUN apt-get install -y curl git sudo
RUN apt-get clean

RUN pip3 install pip --upgrade
RUN pip3 install setuptools --upgrade

RUN mkdir -p /usr/local/pyml
WORKDIR /usr/local/pyml

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt -U