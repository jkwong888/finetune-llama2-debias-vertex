FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

COPY requirements.txt .

RUN pip install -r requirements.txt