FROM python:3.9

ARG GID
ARG UID

RUN echo $GID, $UID
ENV USERNAME=evaluation
ENV GROUPNAME=evalgrp
ENV BASEDIR=/home/evaluation

RUN groupadd -g $GID evalgrp
RUN useradd -m -u $UID -g $GID $USERNAME
RUN mkdir -p $BASEDIR
WORKDIR ${BASEDIR}
RUN apt-get update && apt-get install -y git
RUN chown -R ${USERNAME}:$GROUPNAME ${BASEDIR}
USER ${USERNAME}

RUN git clone https://github.com/openai/human-eval
COPY execpatch.diff human-eval/execpatch.diff
WORKDIR ${BASEDIR}/human-eval
ENV PATH $BASEDIR/.local/bin:$PATH
RUN git checkout 312c5e5
RUN git apply execpatch.diff
RUN pip install -e .
ENTRYPOINT ["evaluate_functional_correctness"]
