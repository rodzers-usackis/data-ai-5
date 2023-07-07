FROM ubuntu:20.04

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "./trainer/EnsembleInTheCloud.py"]