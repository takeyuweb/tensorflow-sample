FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt-get update -qq
RUN apt-get install -y git
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install autopep8

ARG UID=1000
RUN useradd -m -u ${UID} user

ARG WORKDIR=/tensorflow
RUN mkdir -p ${WORKDIR} && chown user:user ${WORKDIR}

USER user
