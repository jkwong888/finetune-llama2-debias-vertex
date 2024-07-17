#!/bin/bash

set +x

TIMESTAMP=$(date +%Y%m%d%H%M%S)
SERVICE_ACCOUNT=notebook@jkwng-vertex-playground.iam.gserviceaccount.com
REGION=us-east1
project=jkwng-vertex-playground

jupyter nbconvert \
 --RegexRemovePreprocessor.patterns="['^\!.*$','.*IPython.*']" \
 quantization-evaluation.ipynb \
 --to python

gcloud ai custom-jobs create \
--region=${REGION} \
--display-name="eval-llama2-${TIMESTAMP}" \
--service-account=${SERVICE_ACCOUNT} \
--args="--model quantized_llama2-7b-chat-hf --version 4 --region ${REGION} --project=${PROJECT}" \
--worker-pool-spec=machine-type=g2-standard-8,replica-count=1,accelerator-type=NVIDIA_L4,executor-image-uri=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime,local-package-path=$(pwd),script=quantization-evaluation.py
