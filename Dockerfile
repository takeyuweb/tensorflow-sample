FROM tensorflow/tensorflow:2.11.0

ARG UID=1000
RUN useradd -m -u ${UID} user

ARG WORKDIR=/tensorflow
RUN mkdir -p ${WORKDIR} && chown user:user ${WORKDIR}

USER user
