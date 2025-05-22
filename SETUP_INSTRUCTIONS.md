# Setup Instructions for a New System

Follow these steps to run the Phi-4 model on a new system after cloning this repository:

## Step 1: Navigate to the Repository Directory

```bash
cd path/to/local-llm-runner
```

## Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

## Step 3: Activate the Virtual Environment

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

## Step 4: Install Required Packages

```bash
pip install -r requirements.txt
```

## Step 5: Hugging Face Authentication

1. Create a Hugging Face account if you don't have one already:
   - Go to https://huggingface.co/join
   - Sign up for an account
   - Accept the model terms at https://huggingface.co/microsoft/phi-4
   - Generate a token at https://huggingface.co/settings/tokens

## Step 6: Run the Phi-4 Model

```bash
python run_phi4.py
```

When prompted, enter your Hugging Face token.

## Step 7: Start Chatting with the Model

Enter your prompts and enjoy!

## Troubleshooting

If you encounter issues with GPU detection or memory, try these alternatives:

- Run with specific parameters:
  ```bash
  python run_phi4.py --temperature 0.8 --max_tokens 1024
  ```

- Force CPU usage (if having GPU issues):
  ```bash
  python run_phi4.py --cpu
  ```

- If you're having memory issues, try the smaller Phi-2 model instead:
  ```bash
  python run_phi2.py
  ```

**Note**: Running on CPU will be much slower than on a GPU. For optimal performance with Phi-4, you'll need a GPU with at least 16GB of VRAM. 