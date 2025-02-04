# DeepSeek Finance LLM Fine-tuning


This repository contains code for fine-tuning the DeepSeek-R1-Distill-Llama-8B model on financial data using LoRA (Low-Rank Adaptation). The implementation uses the Unsloth library for efficient training and inference.

## Overview

The script enables fine-tuning of the DeepSeek model specifically for financial domain tasks using QLoRA (Quantized LoRA) for memory-efficient training. It includes functionality for:

- Loading and quantizing the base model
- Adding LoRA adapters
- Training on a custom financial dataset
- Inference with the fine-tuned model
- Saving and loading the model in various formats (LoRA adapters, GGUF)

## Requirements

```
bitsandbytes
accelerate
xformers==0.0.29
peft
trl
triton
cut_cross_entropy
unsloth_zoo
sentencepiece
protobuf
datasets
huggingface_hub
hf_transfer
unsloth
```

## Model Details

- Base Model: DeepSeek-R1-Distill-Llama-8B
- Quantization: 4-bit quantization
- Maximum Sequence Length: 2048 tokens
- LoRA Configuration:
  - Rank (r): 16
  - Alpha: 16
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0

## Training Configuration

- Batch Size: 2 per device
- Gradient Accumulation Steps: 4
- Learning Rate: 2e-4
- Weight Decay: 0.01
- Scheduler: Linear
- Training Steps: 60
- Warmup Steps: 5
- Optimizer: AdamW 8-bit


[Link for Dataset](https://huggingface.co/datasets/heladell/Finance_DeepSeek-R1-Distill-dataset?row=0)
[Link for GGUF](https://huggingface.co/heladell/Finance_DeepSeek-R1-Distill-Llama-8B_LoRA-GGUF-Q8_0)
