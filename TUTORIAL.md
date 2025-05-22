# Running Local Language Models: A Step-by-Step Guide

This document explains how we set up and run various language models locally, focusing on the Phi-4 model. I'll walk through each component and explain what it does.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Script Components](#script-components)
3. [Code Walkthrough](#code-walkthrough)
4. [Models Overview](#models-overview)
5. [Performance Considerations](#performance-considerations)
6. [Common Issues](#common-issues)

## Environment Setup

We're working within a Python virtual environment (`phi4_env`) which provides isolation for our dependencies. Here's what we've set up:

1. **Python Virtual Environment**: Isolates our project dependencies
2. **Required Libraries**:
   - `torch`: PyTorch for tensor computations and deep learning
   - `transformers`: Hugging Face's library for working with pretrained models
   - `accelerate`: Helps distribute model computations across available hardware
   - `huggingface_hub`: For interacting with Hugging Face's model repository

## Script Components

We created several scripts:

1. `run_phi4.py`: Main script for running the Microsoft Phi-4 model
2. `run_phi2.py`: Script for running the smaller Phi-2 model
3. `run_mistral.py`: Script for running Mistral's 7B model
4. Batch files (`.bat`) for easier execution

## Code Walkthrough

Let's examine the key components of `run_phi4.py`:

### 1. Import Libraries
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, HfFolder
import getpass
import sys
import time
import argparse
```

These imports give us access to:
- PyTorch for tensor operations
- Transformers for model loading and tokenization
- Hugging Face Hub for authentication
- Standard libraries for system operations, timing, and command line arguments

### 2. Argument Parser

```python
def parse_args():
    parser = argparse.ArgumentParser(description="Run Phi-4 model on GPU")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--gpu_layers", type=int, default=None, help="Number of layers to offload to GPU")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision instead of float16")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    return parser.parse_args()
```

This creates command-line options to:
- Force CPU usage
- Control GPU layer allocation
- Select precision type
- Adjust generation parameters

### 3. Hardware Detection

```python
if torch.cuda.is_available() and not args.cpu:
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ VRAM available: {vram:.1f} GB")
    device = "cuda"
else:
    if args.cpu:
        print("Running on CPU (forced by --cpu flag)")
    else:
        print("No GPU detected. Running on CPU (this will be very slow)")
    device = "cpu"
```

This code:
- Checks if a CUDA-compatible GPU is available
- Reports GPU specifications if found
- Falls back to CPU if necessary or if forced

### 4. Authentication

```python
token = HfFolder.get_token()
if token is None:
    print("To use Phi-4, you need to provide your Hugging Face token.")
    # ...
    token = getpass.getpass("\nEnter your Hugging Face token (input will be hidden): ")
    
    try:
        login(token=token)
        print("✓ Authentication successful!")
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        print("Please make sure your token is correct and you've accepted the model terms.")
        return
else:
    print("✓ Using existing Hugging Face authentication")
```

This handles Hugging Face authentication:
- Checks for existing authentication
- Prompts for a token if none found
- Logs in and handles errors

### 5. Model Loading

```python
model_id = "microsoft/phi-4"

# Configure precision
dtype = torch.bfloat16 if args.bf16 else torch.float16

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device_map,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

This code:
- Specifies the model ID from Hugging Face
- Sets precision type (float16 or bfloat16)
- Loads the tokenizer and model with appropriate settings
- Uses `device_map="auto"` to automatically distribute model across available hardware
- Minimizes CPU memory usage

### 6. Chat Loop

```python
messages = [
    {"role": "system", "content": "You are Phi-4, a helpful, harmless, and honest AI assistant..."}
]

while True:
    user_input = input("\n> ")
    if user_input.lower() == 'exit':
        break
    
    # Add user message
    messages.append({"role": "user", "content": user_input})
    
    # Format with chat template
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True,
        repetition_penalty=1.1
    )
    
    # Process and print response...
```

This interactive loop:
- Maintains a conversation history in the `messages` list
- Formats messages using the model's chat template
- Generates responses with the specified parameters
- Extracts and displays the model's responses
- Keeps track of generation time and token counts

### 7. Response Processing

```python
# Get response
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract assistant response
try:
    assistant_response = full_response.split("assistant")[-1].strip()
    if assistant_response.startswith(":"):
        assistant_response = assistant_response[1:].strip()
except:
    assistant_response = full_response

# Add to history
messages.append({"role": "assistant", "content": assistant_response})
```

This extracts the model's response from the generated text:
- Decodes the output tokens to text
- Extracts just the assistant's portion of the response
- Cleans up any formatting artifacts
- Adds the response to the conversation history

## Models Overview

We've set up three models with different characteristics:

1. **Phi-2 (2.7B parameters)**
   - Smaller and faster
   - Works well on CPU
   - Good for basic tasks
   - No authentication required

2. **Phi-4 (14B parameters)**
   - Microsoft's latest model
   - Requires authentication
   - Needs significant computing resources
   - Excellent performance on reasoning tasks

3. **Mistral-7B-Instruct-v0.3 (7B parameters)**
   - Middle ground in size
   - No authentication required
   - Good balance of performance and resource usage
   - Strong instruction-following capabilities

## Performance Considerations

### Hardware Requirements

Model performance depends heavily on available hardware:

1. **CPU Usage**:
   - All models will run on CPU, but slowly
   - Phi-2 is most feasible for CPU-only systems
   - Generation time will be measured in minutes rather than seconds

2. **GPU Requirements**:
   - Phi-2: 4GB+ VRAM
   - Mistral-7B: 8GB+ VRAM
   - Phi-4: 16GB+ VRAM

3. **Memory Optimization Techniques**:
   - Using half-precision (float16/bfloat16) reduces memory requirements
   - `device_map="auto"` distributes model across available hardware
   - `low_cpu_mem_usage=True` minimizes RAM usage during loading

### Generation Parameters

The scripts allow adjusting generation parameters:

- `temperature`: Controls randomness (higher = more creative, lower = more deterministic)
- `max_new_tokens`: Limits response length
- `repetition_penalty`: Reduces repetitive text
- `do_sample`: Enables sampling for more diverse outputs

## Common Issues

1. **Authentication Errors**:
   - Need to accept model terms on Hugging Face
   - Token must have proper permissions
   - Token must be entered correctly

2. **Memory Errors**:
   - "CUDA out of memory" means your GPU doesn't have enough VRAM
   - Solutions: Use a smaller model, reduce precision, or offload layers to CPU

3. **Slow Generation**:
   - CPU generation is much slower than GPU
   - Time metrics show actual performance (e.g., "Generated 90 tokens in 12264.76s")

4. **Model Output Format**:
   - Each model has a specific output format that needs proper extraction
   - Chat history must be maintained in the correct format

## Conclusion

These scripts provide a flexible way to run powerful language models on your local machine. While they require significant resources for optimal performance, they give you complete control over the models without relying on external APIs or services.

For best results:
1. Use a GPU if possible
2. Choose the model size appropriate for your hardware
3. Adjust generation parameters to suit your needs
4. Be patient with generation times, especially on CPU 