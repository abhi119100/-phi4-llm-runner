import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys

def main():
    print("=" * 70)
    print("Mistral-7B Runner")
    print("=" * 70)
    
    # Model ID on Hugging Face - using the openly available version
    model_id = "mistralai/Mistral-7B-v0.1"
    
    print(f"\nLoading {model_id}...")
    print("This model is freely available and doesn't require authentication.")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with progress indicator
        print("Loading model (please be patient)...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Model loaded successfully in {elapsed:.1f} seconds!")
        
        # Check hardware info
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Running on: {device_name} with {vram:.1f} GB VRAM")
        else:
            print("Running on CPU. Inference will be slower.")
        
        # Interactive prompt loop
        print("\n" + "=" * 50)
        print("Chat with Mistral-7B")
        print("=" * 50)
        print("Enter your prompt (or type 'exit' to quit):")
        
        # Initialize with system prompt
        system_prompt = "You are Mistral-7B, a helpful, harmless, and honest AI assistant."
        
        while True:
            user_input = input("\n> ")
            if user_input.lower() == 'exit':
                break
            
            # For base model (not chat-tuned), we need to format the prompt manually
            prompt = f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
            
            # Tokenize and generate
            print("\nGenerating response...")
            gen_start = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            gen_time = time.time() - gen_start
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            try:
                assistant_response = full_response.split("Assistant:")[-1].strip()
            except:
                assistant_response = full_response
            
            # Print response
            print(f"\n{assistant_response}")
            print(f"\n[Generated {len(outputs[0]) - len(inputs.input_ids[0])} tokens in {gen_time:.2f}s]")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. If you're on GPU, ensure you have enough VRAM (at least 8GB recommended)")
        print("3. Make sure you have the transformers and torch libraries installed")
        print("4. Try running with a smaller model if you're experiencing memory issues")

if __name__ == "__main__":
    main() 