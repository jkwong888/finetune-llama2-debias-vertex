#!/bin/bash

set +x

TIMESTAMP=$(date +%Y%m%d%H%M%S)
SERVICE_ACCOUNT=notebook@jkwng-vertex-playground.iam.gserviceaccount.com

gcloud ai custom-jobs create \
--region=us-central1 \
--display-name="quantization-llama2-${TIMESTAMP}" \
--service-account=${SERVICE_ACCOUNT} \
--worker-pool-spec=machine-type=g2-standard-8,replica-count=1,accelerator-type=NVIDIA_L4,executor-image-uri=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime,local-package-path=$(pwd),script=quantization-vertex.py
