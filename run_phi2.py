import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys

def main():
    print("=" * 70)
    print("Phi-2 Model Runner (Lightweight Version)")
    print("=" * 70)
    
    # Model ID on Hugging Face
    model_id = "microsoft/phi-2"
    
    print(f"\nLoading {model_id}...")
    print("This model doesn't require authentication and uses less memory than Phi-4.")
    
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
        print("Chat with Phi-2")
        print("=" * 50)
        print("Enter your prompt (or type 'exit' to quit):")
        
        history = ""
        system_prompt = "You are Phi-2, a helpful, harmless, and honest AI assistant developed by Microsoft."
        
        while True:
            user_input = input("\n> ")
            if user_input.lower() == 'exit':
                break
            
            # Create a chat-like format
            if not history:
                # First message
                prompt = f"System: {system_prompt}\n\nHuman: {user_input}\n\nAssistant:"
            else:
                # Subsequent messages
                prompt = f"{history}\n\nHuman: {user_input}\n\nAssistant:"
            
            # Prepare the input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            print("\nGenerating response...")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            assistant_response = full_response.split("Assistant:")[-1].strip()
            
            # Update history
            history = f"{prompt} {assistant_response}"
            
            # Print response
            print(f"\n{assistant_response}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. If you're on GPU, ensure you have enough VRAM")
        print("3. Make sure you have the transformers and torch libraries installed")

if __name__ == "__main__":
    main() 