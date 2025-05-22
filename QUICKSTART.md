# Local LLM Runner: Quick Start Guide

This is a simplified guide to get you started with running local language models.

## Quick Setup

1. **Prerequisites**:
   - Python 3.8+ installed
   - Sufficient RAM (8GB minimum, 16GB+ recommended)
   - GPU with CUDA support (optional but recommended)

2. **Install required packages**:
   ```
   pip install torch transformers accelerate huggingface_hub
   ```

3. **Accept model terms (for Phi-4 only)**:
   - Create a Hugging Face account: https://huggingface.co/join
   - Visit https://huggingface.co/microsoft/phi-4 and accept the terms
   - Generate a token at https://huggingface.co/settings/tokens

## Running the Models

### Option 1: Phi-2 (Smallest, Fastest)
```
python run_phi2.py
```
or
```
.\run_phi2.bat
```

### Option 2: Mistral-7B (Medium Size, Good Quality)
```
python run_mistral.py
```
or
```
.\run_mistral.bat
```

### Option 3: Phi-4 (Largest, Best Quality)
```
python run_phi4.py
```
or
```
.\run_phi4_gpu.bat
```

## Advanced Options (Phi-4)

Control how Phi-4 runs with command-line arguments:

```
python run_phi4.py --temperature 0.8 --max_tokens 1024
```

Available options:
- `--cpu`: Force CPU usage even if GPU is available
- `--bf16`: Use bfloat16 precision instead of float16
- `--temperature`: Set creativity level (0.7 default)
- `--max_tokens`: Maximum tokens to generate (512 default)

## Using the Chat Interface

1. Enter your prompt after the `>` symbol
2. Wait for the model to generate a response
   - CPU generation may take minutes
   - GPU generation should be faster
3. Continue the conversation with follow-up prompts
4. Type `exit` to quit

## Troubleshooting

### Common Issues:

1. **Slow generation**: Normal on CPU; shown as "[Generated X tokens in Y seconds]"
2. **Out of memory errors**: Try a smaller model or reduce parameters
3. **Authentication errors**: Make sure you've accepted model terms and entered correct token

### Model Selection Guide:

| Model | Size | Memory Needed | Authentication | Best For |
|-------|------|---------------|----------------|----------|
| Phi-2 | 2.7B | 4GB+ | No | Quick responses, basic tasks |
| Mistral-7B | 7B | 8GB+ | No | General tasks, balanced speed/quality |
| Phi-4 | 14B | 16GB+ | Yes | Complex reasoning, highest quality |

Choose the model that best fits your hardware and needs. 