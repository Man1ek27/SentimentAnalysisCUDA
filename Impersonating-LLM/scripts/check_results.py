#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_ADAPTER = "./checkpoints/trump_qwen_lora/best_adapter"

def parse_args():
    p = argparse.ArgumentParser(description="Generate Trump-style tweets using a fine-tuned LoRA adapter")
    p.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER,
                   help="Path to the saved LoRA adapter folder")
    p.add_argument("--prompt", type=str, default="The fake news",
                   help="The initial text/prompt to complete")
    p.add_argument("--max-tokens", type=int, default=60,
                   help="Maximum number of new tokens to generate")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the base model (using fp16 to save VRAM on GPU)
    print("Loading base model...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map={"": device}
    )

    # 3. Load and apply the LoRA adapter
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model.config.pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        print(f"\nError loading adapter: {e}")
        print("Make sure the adapter path is correct and contains adapter_model.safetensors.")
        return

    model.eval()
    print(f"Prompt: '{args.prompt}'")
    print("\nGenerating...\n")

    # 4. Tokenize prompt and generate text
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_tokens,
            do_sample=True,          # Enable sampling for creative generation
            top_k=50,                # Restrict sampling pool to top 50 tokens
            top_p=0.95,              # Nucleus sampling (cumulative probability cutoff)
            temperature=0.8,         # Higher temperature equals more stylized/creative output
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15, # Penalize repeating identical phrases
            no_repeat_ngram_size=3   # Block repeating any 3-token sequences
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Result:")
    print(f"  {generated_text}\n")

if __name__ == "__main__":
    main()
