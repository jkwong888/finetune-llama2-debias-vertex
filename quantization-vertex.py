#!/usr/bin/env python
# coding: utf-8

# # Quantization Demo

# ## Introduction
# In this demo, we will employ PEFT (LoRA) and Quantization techniques to fine-tune the Llama2-7b model, aiming to debias and detoxify text. We will utilize a specific dataset located at `../../data/debiased_profanity_check_with_keywords.csv`.
# 
# This notebook will guide you through the process, showcasing the steps involved in fine-tuning the model to produce a debiased and detoxified output from biased or toxic text.

# ## Steps
# 
# Here we define the main steps to fine-tune the Llama2-7b model using QLoRA.
# 
# 1.   Load the dataset and apply necessary transformations to format it for prompt-completion.
# 2.   Configure bitsandbytes for 4-bit quantization; define the load and compute data types as specified in the QLoRA paper.
# 3.   Load the LlaMA2 model and its tokenizer.
# 4.   Define LoRA configurations and Training Arguments.
# 5.   Train using the SFT Trainer, which by default stores only the adapter model.
# 6.   Merge the adapter model with the base model (loaded in FP16).

# ## Importing Libraries
# This cell imports libraries for dataset loading, tokenization, and training large language models using Hugging Face Transformers, and libraries required for PEFT and quantization.
# 
# 

# In[ ]:


import os
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer


# ## Configuring Directory Paths for Model Weights, Dataset, and Model Storage
# This cell specifies the directory paths for storing model checkpoints, adapter models, merged models, and the dataset necessary for the task.

# In[ ]:


bucket_name = "jkwng-llama-experiments"
model_dir = "llama2-7b-chat-hf"
model_bucket_prefix = "llama2"
model_path = f"gs://{bucket_name}/{model_bucket_prefix}/{model_dir}"


# In[ ]:


#DATASET_PATH = "../../data/debiased_profainty_check_with_keywords.csv" # dataset of biased and corresponding debiased text
DATASET_PATH = f"gs://{bucket_name}/debiased_profainty_check_with_keywords.csv"
OUTPUT_DIR = "projects/fta_bootcamp/quantization/" # main directory of the the demo output
CHECKPOINT_DIR = f"{OUTPUT_DIR}checkpoint" # where to save checkpoints
MERGED_MODEL_DIR= f"{OUTPUT_DIR}merged_model"  # where to save merged model


# In[ ]:


local_model_dir = "projects/fta_bootcamp/downloads"
MODEL_NAME = f"{local_model_dir}/Llama-2-7b-chat-hf" # chat model
NEW_MODEL_NAME = "llama-2-7b-debiaser" # Fine-tuned model name


# In[ ]:


from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

os.makedirs(MODEL_NAME, exist_ok=True)

#print(list(bucket.list_blobs(prefix=f"{model_bucket_prefix}/{model_dir}")))

# download all files from path
for blob in bucket.list_blobs(prefix=f"{model_bucket_prefix}/{model_dir}"):
  print(f"downloading {blob.name} to {MODEL_NAME}/{os.path.basename(blob.name)}")
  blob.download_to_filename(f"{MODEL_NAME}/{os.path.basename(blob.name)}")


# ## Creating a HuggingFace Dataset

# In[ ]:


def create_hf_dataset_from_csv(csv_path):
  dataset = load_dataset('csv', data_files=csv_path, split='train')
  return dataset

dataset = create_hf_dataset_from_csv(DATASET_PATH)
dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.select_columns(["biased_text", "debiased_text"])


# Here are the first 3 samples of the dataset:

# In[ ]:


print(len(dataset["train"]))
print(len(dataset["test"]))

for i in range(3):
    sample = dataset["train"][i]
    print(sample, '\n')


# Write the datasets to storage

# In[ ]:


# write the train and test dataset to disk
os.makedirs(os.path.join(MERGED_MODEL_DIR, "dataset"), exist_ok=True)

dataset["train"].save_to_disk(f"{MERGED_MODEL_DIR}/dataset/train")
dataset["test"].save_to_disk(f"{MERGED_MODEL_DIR}/dataset/test")


# ## Loading Tokenizer

# In[ ]:


tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, add_eos_token=True)

if not tokenizer.pad_token:
  tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.model_max_length = 1024



# ## Formatting Prompts
# For instruction fine-tuning, we will use Stanford-Alpaca format as follows:
# 
# `### Instruction:\n {prompt}\n ### Input:\n {input_text}\n ### Response\n: {completion}`

# In[ ]:


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


# ## Configuring Quantization and LoRA

# ### LoRA-Specific Parameters
# 
# *   r: Rank is essentially a measure of how the original weight matrices are broken down into simpler, smaller matrices.
# *   lora_alpha: Alpha parameter for LoRA scaling. This parameter controls the scaling of the low-rank approximation. Higher values might make the approximation more influential in the fine-tuning process, affecting both performance and computational cost.
# *   lora_dropout: Dropout probability for LoRA layers. This is the probability that each neuron’s output is set to zero during training, used to prevent overfitting.
# 
# https://arxiv.org/abs/2305.14314

# In[ ]:


peft_config = LoraConfig(
  r=64,
  lora_alpha=16, # Alpha parameter for LoRA scaling. This parameter controls the scaling of the low-rank approximation. Higher values might make the approximation more influential in the fine-tuning process, affecting both performance and computational cost.
  lora_dropout=0.2, # Dropout probability for LoRA layers. This is the probability that each neuron’s output is set to zero during training, used to prevent overfitting.
  bias="none",
  task_type="CAUSAL_LM"
)


# ### Quantization Parameters
# 
# We utilize ****4bit quantization**** as described in the QLoRA paper : https://arxiv.org/pdf/2305.14314.pdf
# 
# QLoRA paper sets parameters as follows:
# 
# * set load_in_4bit=True to quantize the model to 4-bits when you load it.
# * set bnb_4bit_quant_type="nf4" to use a special 4-bit data type for weights initialized from a normal distribution.
# * set bnb_4bit_use_double_quant=True to use a nested quantization scheme to quantize the already quantized weights.
# * set bnb_4bit_compute_dtype=torch.bfloat16 to use bfloat16 for faster computation.

# In[ ]:


use_4bit = True # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16" # Compute dtype for 4-bit base models : either float16 or bfloat16, bfloat16 is recommended as it produces less nans ** Note bnb_4bit_compute_dtype for merging.
bnb_4bit_quant_type = "nf4" # Quantization type (fp4 or nf4)
use_nested_quant = False # Activate nested quantization for 4-bit base models (double quantization)
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)


# ## Loading Model

# In the cell below, we create a model object with the defined quantizaition configuration from the bitsandbytes library

# In[ ]:


device_map = {"":0}
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map=device_map,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1 # Setting this to a value different than 1 will activate the more accurate but slower computation of the linear layers, which should better match the original logits.


# ### Base Model Generation
# Here we test the performance of the base model:

# In[ ]:


instruction = (
    " You are a text debiasing bot, you take as input a"
    " text and you output its debiased version by rephrasing it to be"
    " free from any age, gender, political, social or socio-economic"
    " biases, without any extra outputs. Debias this text by rephrasing"
    " it to be free of bias: "
)

input_text = "Women are dumb."
text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}
'''

pipe = pipeline('text-generation', model=base_model, tokenizer=tokenizer, max_length=400)
result = pipe(text)
result[0]['generated_text']


# ### Calculating Trainable Parameters of the Model

# In[ ]:


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(base_model)


# ## Defining Training Arguments

# In[ ]:


training_arguments = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=20000,
    logging_steps=250,
    logging_dir='./logs',
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)
model = prepare_model_for_kbit_training(base_model)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func = formatting_prompts_func,
    packing=False,
)


# ## Training the Model

# In[ ]:


trainer.train()


# ## Merge the Model

# In[ ]:


model = trainer.model.save_pretrained(os.path.join(MERGED_MODEL_DIR, "adapter")) # save adapter weights
#trainer.model.config.to_json_file(os.path.join(MERGED_MODEL_DIR, "adapter", "adapter_config.json"))

del model
del base_model
del trainer


# In[ ]:


import torch
from datetime import datetime
import os
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
)
from peft import PeftModel

local_model_dir = "projects/fta_bootcamp/downloads"
MODEL_NAME = f"{local_model_dir}/Llama-2-7b-chat-hf" # chat model
OUTPUT_DIR = "projects/fta_bootcamp/quantization/" # main directory of the the demo output
MERGED_MODEL_DIR= f"{OUTPUT_DIR}merged_model"  # where to save merged model
bucket_name = "jkwng-llama-experiments"
model_bucket_prefix = "llama2"
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_dir = "llama2-7b-chat-hf"

output_model_parent = f"quantized_{model_dir}"
output_model_version = timestamp
output_model_name = f"{output_model_parent}_{output_model_version}"
output_model_path = f"{model_bucket_prefix}/{output_model_name}"
output_model_full_path = f"gs://{bucket_name}/{output_model_path}"



# In[ ]:


tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, add_eos_token=True)

if not tokenizer.pad_token:
  tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.model_max_length = 1024

device_map = {"":0}
# reload base model at half precision
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
    device_map=device_map,
    torch_dtype=torch.float16,
)

# merge the adapter weights
peft_model = PeftModel.from_pretrained(
    model=base_model,
    model_id=os.path.join(MERGED_MODEL_DIR, "adapter"),
    local_files_only=True,
)


# In[ ]:


merged_model = peft_model.merge_and_unload(progressbar=True)
print(base_model)
print(merged_model)


# In[ ]:


merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=False)
tokenizer.save_pretrained(MERGED_MODEL_DIR)


# ## Publish the Model

# ### Write the model weights to cloud storage

# In[ ]:


from pathlib import Path
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# First, recursively get all files in `directory` as Path objects.
directory_as_path_obj = Path(MERGED_MODEL_DIR)
paths = directory_as_path_obj.rglob("*")

# Filter so the list only includes files, not directories themselves.
file_paths = [path for path in paths if path.is_file()]

# These paths are relative to the current working directory. Next, make them
# relative to `directory`
relative_paths = [path.relative_to(MERGED_MODEL_DIR) for path in file_paths]

# Finally, convert them all to strings.
string_paths = [str(path) for path in relative_paths]

# Start the upload.
for path in file_paths:
  relative_path = path.relative_to(MERGED_MODEL_DIR)
  blob = bucket.blob(f"{output_model_path}/{str(relative_path)}")
  print(f"uploading {path.stat().st_size} bytes {str(path)} to {blob.name}")
  blob.upload_from_filename(str(path))




# ### publish the model to  Vertex AI

# In[ ]:


from google.cloud import aiplatform

aiplatform.init()


# In[ ]:


from platform import version
import google.api_core.exceptions

VLLM_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240326_0916_RC00"

vertex_model = None
vllm_args = [
  "--host=0.0.0.0",
  "--port=7080",
  f"--tensor-parallel-size=1",
  "--swap-space=4",
  "--gpu-memory-utilization=0.9",
  f"--max-model-len=4096",
  f"--dtype=float16",
  "--disable-log-stats",
]

env_vars = {
    "MODEL_ID": output_model_full_path,
    "DEPLOY_SOURCE": "notebook",
}

# Publish the model to the Vertex Model Registry
try:
  vertex_model = aiplatform.Model(model_name=f"{output_model_parent}")

  print(f"Model: {vertex_model.to_dict()}")

  # publish new version
  vertex_model = vertex_model.upload(
      display_name=f"{output_model_parent}",
      version_aliases=[f"v{output_model_version}"],
      parent_model=f"{output_model_parent}",
      artifact_uri=output_model_full_path,
      serving_container_image_uri=VLLM_DOCKER_URI,
      serving_container_command=["python", "-m", "vllm.entrypoints.api_server"],
      serving_container_args=vllm_args,
      serving_container_ports=[7080],
      serving_container_predict_route="/generate",
      serving_container_health_route="/ping",
      serving_container_environment_variables=env_vars,
      serving_container_shared_memory_size_mb=(4 * 1024),  # 4 GB
      serving_container_deployment_timeout=7200,

  )
except google.api_core.exceptions.NotFound as e:
    print("Model not found. Creating new model...")

    vertex_model = aiplatform.Model.upload(
      display_name=f"{output_model_parent}",
      model_id=f"{output_model_parent}",
      version_aliases=[f"v{output_model_version}"],
      artifact_uri=output_model_full_path,
      serving_container_image_uri=VLLM_DOCKER_URI,
      serving_container_command=["python", "-m", "vllm.entrypoints.api_server"],
      serving_container_args=vllm_args,
      serving_container_ports=[7080],
      serving_container_predict_route="/generate",
      serving_container_health_route="/ping",
      serving_container_environment_variables=env_vars,
      serving_container_shared_memory_size_mb=(4 * 1024),  # 4 GB
      serving_container_deployment_timeout=7200,
    )

print(f"Vertex model: {vertex_model.to_dict()}")


# ## Load and Test Trained Model

# In[ ]:


from vllm import LLM, SamplingParams
llm = LLM(model=MERGED_MODEL_DIR)


# ### Trained Model Generation
# Here we test the performance of the trained model:

# In[ ]:


instruction = (
    " You are a text debiasing bot, you take as input a"
    " text and you output its debiased version by rephrasing it to be"
    " free from any age, gender, political, social or socio-economic"
    " biases, without any extra outputs. Debias this text by rephrasing"
    " it to be free of bias: "
)

input_text = "Women are dumb."
text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}
'''
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt!r}, Generated text: {generated_text!r}")

