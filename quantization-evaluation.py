#!/usr/bin/env python
# coding: utf-8

# # LLM Evaluation Demo

# ## Introduction
# In this demo, we will evaluate how well our PEFT (LoRA) and Quantization techniques to fine-tune the Llama2-7b model, aiming to debias and detoxify text. We will utilize a specific dataset located at `../../data/debiased_profanity_check_with_keywords.csv`.
# 
# This notebook will guide you through the process, showcasing the steps involved in evaluating the fine-tuned model to produce a debiased and detoxified output from biased or toxic text, comparing it to the ground truth stored in storage.

# ## Importing Libraries
# This cell imports libraries for dataset loading.
# 
# 

# In[59]:


import os
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    logging,
)

from google.cloud import aiplatform
from pprint import pprint

import argparse


# ## Configuring Directory Paths for Model Weights, Dataset, and Model Storage
# This cell specifies the directory paths for storing model checkpoints, adapter models, merged models, and the dataset necessary for the task. We will load the test set to perform inference against the finetuned model.

# In[3]:


# Get the default cloud project id.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

# Get the default region for launching jobs.
REGION = os.getenv("GOOGLE_CLOUD_REGION")

output_model_parent = "quantized_llama2-7b-chat-hf"
output_model_version = 4


# In[64]:


parser = argparse.ArgumentParser(
                    prog='quantization-evaluation',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('-m', '--model', default=output_model_parent, required=False)
parser.add_argument('-v', '--version', default=output_model_version, required=False)
parser.add_argument('-r', '--region', default=REGION, required=False)
parser.add_argument('-p', '--project', default=PROJECT_ID, required=False)
args, unknown = parser.parse_known_args()


# In[66]:


#REGION = "us-central1"
aiplatform.init(location=args.region)

from google.cloud import aiplatform_v1

model_registry = aiplatform.models.ModelRegistry(model=args.model)

# Get model version info with the version 'version_id'.
model_version_info = model_registry.get_version_info(version=args.version)
pprint(model_version_info)
print(f"projects/{args.project}/locations/{args.region}/models/{model_version_info.model_display_name}@{model_version_info.version_id}")

model = aiplatform.Model(model_name=args.model, version=args.version)
model_dict = model.to_dict()
pprint(model_dict)

model_path = model_dict['artifactUri']


# # Download the artifacts
# 
# Download the model weights and datasets to a local directory.

# In[67]:


from google.cloud import storage

bucket_name = model_path.split("gs://")[1].split("/")[0]
bucket_prefix = str.join('/', model_path.split("gs://")[1].split("/")[1:])
print(f"{bucket_name}, {bucket_prefix}")

local_model_path = f"models/{model_version_info.model_display_name}"

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

os.makedirs(local_model_path, exist_ok=True)

#print(list(bucket.list_blobs(prefix=f"{model_bucket_prefix}/{model_dir}")))

# download all files from path
for blob in bucket.list_blobs(prefix=f"{bucket_prefix}/"):
  local_blob_name = str.join('/', blob.name.split(bucket_prefix)[1].split("/")[1:])
  os.makedirs(os.path.dirname(f"{local_model_path}/{local_blob_name}"), exist_ok=True)

  if os.path.exists(f"{local_model_path}/{local_blob_name}"):
    continue

  print(f"downloading {blob.name} to {local_model_path}/{local_blob_name}")
  #print(blob.name)
  blob.download_to_filename(f"{local_model_path}/{local_blob_name}")


# ## Creating a HuggingFace Dataset

# Load the test dataset from storage

# In[51]:


#DATASET_PATH = "../../data/debiased_profainty_check_with_keywords.csv" # dataset of biased and corresponding debiased text
DATASET_PATH = f"{local_model_path}/dataset/test/"

def create_hf_dataset_from_csv(csv_path):
  dataset = load_dataset('arrow', data_files=csv_path, split="train")
  return dataset

dataset = create_hf_dataset_from_csv(f"{DATASET_PATH}data-00000-of-00001.arrow")
#dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.select_columns(["biased_text", "debiased_text"])


# Here are the first 3 samples of the dataset:

# In[52]:


print(len(dataset))

for i in range(3):
    sample = dataset[i]
    print(sample, '\n')


# ## Loading Tokenizer

# In[8]:


tokenizer = LlamaTokenizer.from_pretrained(local_model_path, trust_remote_code=True, add_eos_token=True)

if not tokenizer.pad_token:
  tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.model_max_length = 1024



# In[9]:


def formatting_prompts_func(examples):
    instruction = (
        " You are a text debiasing bot, you take as input a"
        " text and you output its debiased version by rephrasing it to be"
        " free from any age, gender, political, social or socio-economic"
        " biases, without any extra outputs. Debias this text by rephrasing"
        " it to be free of bias: "
    )
    output_text = []
    for i in range(len(examples["biased_text"])):
        input_text = examples["biased_text"][i]
        response = examples["debiased_text"][i]

        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input_text}

        ### Response:
        {response}
        '''

        output_text.append(text)

    return output_text


# # Load and Test Trained Model

# ## Trained Model Generation
# Here we test the performance of the trained model. Load the trained model to GPU memory:

# In[10]:


from vllm import LLM, SamplingParams
llm = LLM(model=local_model_path)


# Set up some common sampling parameters.

# In[48]:


max_tokens = 256
temperature = 1.0
top_p = 0.9
top_k = 1
sampling_params = SamplingParams(
    temperature=temperature,
    max_tokens=max_tokens,
    top_k=top_k,
    top_p=top_p,
    n=1,
    stop=["        ", "\n\n\n"])


# ## Run offline inference
# 
# Use the dataset to generate prompts to generate the array of predictions we will use to compare to ground truth.

# In[53]:


instruction = (
    " You are a text debiasing bot, you take as input a"
    " text and you output its debiased version by rephrasing it to be"
    " free from any age, gender, political, social or socio-economic"
    " biases, without any extra outputs. Debias this text by rephrasing"
    " it to be free of bias: "
)

prompts = []
predictions = []
references = []
for data in dataset:
  #print(data)
  input_text = data["biased_text"]
  expected = data["debiased_text"]
  references.append(expected)
  #print(input_text)
  text = f'''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
  '''

  prompts.append(text)

outputs = llm.generate(prompts, sampling_params)
for response in outputs:
  #print(len(response.outputs))
  #print(response.outputs[0].text)
  output_pred = response.outputs[0].text.strip()
  predictions.append(output_pred)
  print(f"output: {output_pred}")

  #print(f"expected: {expected}")


# # Calculate Rouge Score
# 
# use the Huggingface `evaluate` library to calculate the aggregate ROUGE score.

# In[54]:


# prompt: calculate rouge score

import evaluate

rouge = evaluate.load("rouge", trust_remote_code=True)
results = rouge.compute(predictions=predictions,
                        references=references)
print(results)


# In[68]:


from google.cloud.aiplatform import gapic

metrics = {
  "rougeLSum": results['rougeLsum']
}

now = datetime.now().strftime("%Y%m%d%H%M%S")
eval_name = f"eval_{now}"
model_eval = gapic.ModelEvaluation(
    display_name=eval_name,
    metrics_schema_uri="gs://google-cloud-aiplatform/schema/modelevaluation/general_text_generation_metrics_1.0.0.yaml",
    metrics=metrics,
)

print(model)
API_ENDPOINT = f"{args.region}-aiplatform.googleapis.com"
client = gapic.ModelServiceClient(client_options={"api_endpoint": API_ENDPOINT})
client.import_model_evaluation(
  parent=f"projects/{args.project}/locations/{args.region}/models/{model_version_info.model_display_name}@{model_version_info.version_id}",
  model_evaluation=model_eval,
)

print(f"wrote evaluation \"{eval_name}\" to model {model_version_info.model_display_name}@{model_version_info.version_id}")

