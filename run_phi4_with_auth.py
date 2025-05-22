import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, HfFolder
import os
import getpass
import sys
import time

def check_auth_status():
    """Check if user is already logged in to Hugging Face"""
    token = HfFolder.get_token()
    return token is not None

def main():
    print("=" * 70)
    print("Phi-4 Model Runner")
    print("=" * 70)
    
    # Check if already logged in
    if check_auth_status():
        print("✓ You're already logged in to Hugging Face.")
    else:
        print("To access Phi-4, you need to:")
        print("1. Create a Hugging Face account if you don't have one: https://huggingface.co/join")
        print("2. Accept the model terms at: https://huggingface.co/microsoft/phi-4")
        print("3. Generate a token at: https://huggingface.co/settings/tokens")
        print("\nPlease provide your Hugging Face token to proceed:")
        token = getpass.getpass("Enter your Hugging Face token (input will be hidden): ")
        
        # Login to Hugging Face
        try:
            login(token=token)
            print("✓ Successfully logged in to Hugging Face.")
        except Exception as e:
            print(f"❌ Authentication failed: {str(e)}")
            print("Please make sure you have:")
            print("- Created a Hugging Face account")
            print("- Generated a valid token")
            print("- Accepted the model terms at https://huggingface.co/microsoft/phi-4")
            sys.exit(1)
    
    # Model ID on Hugging Face
    model_id = "microsoft/phi-4"
    
    print(f"\nLoading {model_id}...")
    print("This might take several minutes depending on your internet connection and hardware.")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with progress indicator
        print("Loading model (please be patient)...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto",  # Automatically decide device placement
            trust_remote_code=True
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Model loaded successfully in {elapsed:.1f} seconds!")
        
        # Check hardware info
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Running on: {device_name} with {vram:.1f} GB VRAM")
        else:
            print("Running on CPU. Inference will be slower.")
        
        # Interactive prompt loop
        print("\n" + "=" * 50)
        print("Chat with Phi-4")
        print("=" * 50)
        print("Enter your prompt (or type 'exit' to quit):")
        
        # System prompt
        system_prompt = "You are Phi-4, a helpful, harmless, and honest AI assistant developed by Microsoft."
        messages = [{"role": "system", "content": system_prompt}]
        
        while True:
            user_input = input("\n> ")
            if user_input.lower() == 'exit':
                break
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Create input for the model using the messages list
            inputs = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            print("\nGenerating response...")
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            assistant_response = full_response.split("assistant")[-1].strip()
            if assistant_response.startswith(":"):
                assistant_response = assistant_response[1:].strip()
            
            # Add assistant message to messages list
            messages.append({"role": "assistant", "content": assistant_response})
            
            # Print response
            print(f"\n{assistant_response}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Make sure you've accepted the model terms at https://huggingface.co/microsoft/phi-4")
        print("2. Check your internet connection")
        print("3. If you're on GPU, ensure you have enough VRAM (at least 16GB recommended)")
        print("4. Try running with smaller model like 'microsoft/phi-2' which requires less memory")

if __name__ == "__main__":
    main() 