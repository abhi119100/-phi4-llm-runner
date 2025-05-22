import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, HfFolder
import getpass
import sys
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Phi-4 model on GPU")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--gpu_layers", type=int, default=None, help="Number of layers to offload to GPU (None = all)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision instead of float16")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("Phi-4 Model Runner (GPU Accelerated)")
    print("=" * 70)
    
    # Check if CUDA is available
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
    
    # Check for Hugging Face token
    token = HfFolder.get_token()
    if token is None:
        print("To use Phi-4, you need to provide your Hugging Face token.")
        print("1. Create a Hugging Face account: https://huggingface.co/join")
        print("2. Accept the model terms: https://huggingface.co/microsoft/phi-4")
        print("3. Generate a token: https://huggingface.co/settings/tokens")
        token = getpass.getpass("\nEnter your Hugging Face token (input will be hidden): ")
        
        # Login to Hugging Face
        try:
            login(token=token)
            print("✓ Authentication successful!")
        except Exception as e:
            print(f"❌ Authentication failed: {str(e)}")
            print("Please make sure your token is correct and you've accepted the model terms.")
            return
    else:
        print("✓ Using existing Hugging Face authentication")
    
    # Model ID
    model_id = "microsoft/phi-4"
    
    print(f"\nLoading {model_id}...")
    print("This may take a few minutes depending on your hardware.")
    
    try:
        # Configure GPU memory usage
        dtype = torch.bfloat16 if args.bf16 else torch.float16
        
        # Determine device map
        if device == "cuda" and args.gpu_layers is not None:
            # Custom device map - partial GPU offloading
            print(f"Using {args.gpu_layers} layers on GPU")
            device_map = "auto"
        else:
            # Full auto mapping
            device_map = "auto"
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model
        print("Loading model to GPU (please be patient)...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Model loaded successfully in {elapsed:.1f} seconds!")
        print(f"✓ Using {dtype} precision")
        
        # Chat loop
        print("\n" + "=" * 50)
        print("Chat with Phi-4")
        print("=" * 50)
        print("Enter your prompt (or type 'exit' to quit):")
        
        # Initialize chat
        messages = [
            {"role": "system", "content": "You are Phi-4, a helpful, harmless, and honest AI assistant developed by Microsoft."}
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
            
            # Generate
            print("\nGenerating response...")
            gen_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            gen_time = time.time() - gen_start
            
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
            
            # Display
            print(f"\n{assistant_response}")
            print(f"\n[Generated {len(outputs[0]) - len(inputs[0])} tokens in {gen_time:.2f}s]")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you've accepted the model terms at https://huggingface.co/microsoft/phi-4")
        print("2. Check your internet connection")
        print("3. Ensure you have enough VRAM (16GB+ recommended)")
        print("4. Try using 'microsoft/phi-2' instead if you have limited resources")
        print("5. If running out of memory, try with --gpu_layers option to use fewer layers on GPU")
        print("6. If on Windows, consider running in WSL for better GPU performance")

if __name__ == "__main__":
    main() 