FROM python:3.9

ENV USERNAME=evaluation
ENV BASEDIR=/cgeval

WORKDIR ${BASEDIR}
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install numpy==1.21.2 psutil==5.9.4
RUN groupadd -r ${USERNAME} && useradd --no-log-init -r -g ${USERNAME} ${USERNAME} 
RUN chown -R ${USERNAME}:${USERNAME} ${BASEDIR}
USER ${USERNAME}

RUN git clone https://github.com/salesforce/CodeGen.git
WORKDIR ${BASEDIR}/CodeGen
RUN git checkout fe6a3f1
WORKDIR ${BASEDIR}/CodeGen/codegen1/benchmark
ENTRYPOINT ["python", "mtpb_exec.py"]
