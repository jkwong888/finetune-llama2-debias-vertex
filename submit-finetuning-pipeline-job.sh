#!/bin/bash

set +x

TIMESTAMP=$(date +%Y%m%d%H%M%S)
SERVICE_ACCOUNT=notebook@jkwng-vertex-playground.iam.gserviceaccount.com
REGION=us-east1
PROJECT=jkwng-vertex-playground
TRAINER_IMAGE=gcr.io/jkwng-vertex-playground/cloudai/eval-llama2
#TRAINER_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
MODEL_ID=quantized_llama2-7b-chat-hf


jupyter nbconvert \
 --RegexRemovePreprocessor.patterns="['^\!.*$','.*IPython.*']" \
 quantization-vertex.ipynb \
 --to python

gcloud ai custom-jobs create \
--region=${REGION} \
--display-name="finetune-llama2-${TIMESTAMP}" \
--service-account=${SERVICE_ACCOUNT} \
--args="--model ${MODEL_ID} --region ${REGION} --project=${PROJECT}" \
--worker-pool-spec=machine-type=g2-standard-8,replica-count=1,accelerator-type=NVIDIA_L4,executor-image-uri=${TRAINER_IMAGE},local-package-path=$(pwd),script=quantization-vertex.py
