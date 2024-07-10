# Fine tuning Llama 2 on Vertex AI Pipelines

An example notebook on Finetuning Llama2 chat to debias texts, using the [dataset](https://huggingface.co/datasets/newsmediabias/debiased_dataset) hosted on Huggingface. This was provided as an example on finetuning Llama2, but stopped short of deployment and executing in a full pipeline.

We updated the notebook to save and load datasets and model weights to Google Cloud Storage, and publish the model to the Vertex AI model registry.

## Finetuning notebook

Use the attached notebook [quantization-vertex.ipynb](quantization-vertex.ipynb) in Colab Enterprise to fine tune the model interactively.

Convert it to a python script using [nbconvert](https://nbconvert.readthedocs.io/en/latest/):

```
jupyter nbconvert --RegexRemovePreprocessor.patterns="['^\!.*$','.*IPython.*']" quantization-vertex.ipynb --to python
```

Note we had to remove shell commands from the notebook manually as they don't work outside of the notebook, and remove the IPython commands as we are running the python code outside of a notebook.

You can use the [submit-finetuning-pipeline-job.sh](submit-finetuning-pipeline-job.sh) bash script to submit it as a job on Vertex AI. We use the `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` script and install the dependencies in [requirements.txt](requirements.txt) to run a custom job in Vertex AI on a `g2-standard-8` machine with a single L4 GPU. 

## Deploy the model to Vertex AI Endpoint for prediction

The model can be deployed to a serving container (we used vLLM) for online prediction.  Please view the notebook [quantization-inference.ipynb](quantization-inference.ipynb) for details.


## Run an evaluation in a pipeline

We can compare ground truth biased/debiased text pairs to outputs of our prediction and calculate an aggregate RougeL score that represents how good our model is at the task we tuned it on.  We can then publish this to the model version in the Vertex AI model registry to keep track of model performance.  Please see the notebook [quantization-evaluation.ipynb](quantization-evaluation.ipynb) for details. 

We can convert this to a custom job on Vertex AI using the test set that we set aside during the finetuning process.

(This part is still a work in progress).