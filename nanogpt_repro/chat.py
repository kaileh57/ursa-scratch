import torch
import os
from train_gpt2 import GPT, GPTConfig
import tiktoken
import argparse

def chat(args):
    """
    An interactive chat script to talk to the trained GPT-2 model.
    """
    # --- Load the model from checkpoint ---
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Re-create the model from the saved configuration
    config_args = checkpoint['config']
    config = GPTConfig(**vars(config_args))
    model = GPT(config)
    
    # Load the state dictionary
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    model.to(args.device)
    print("Model loaded successfully.")

    # --- Set up the tokenizer ---
    enc = tiktoken.get_encoding("gpt2")

    # --- Start the interactive chat loop ---
    print("\n--- Start Chatting (type 'exit' or 'quit' to end) ---")
    
    # Initial context is just the end-of-text token to start generation
    context_tokens = [enc.eot_token]

    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        # Encode user input and add to the context
        user_tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        context_tokens.extend(user_tokens)
        
        # Convert context to a tensor
        context_tensor = torch.tensor([context_tokens], dtype=torch.long, device=args.device)

        # Truncate context if it exceeds block size
        context_tensor = context_tensor[:, -config.block_size:]

        # --- Generate the model's response ---
        print("Bot:", end="", flush=True)
        
        # Use the model's generate function
        # We'll generate tokens one by one and stream them to the console
        with torch.no_grad():
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                for _ in range(args.max_new_tokens):
                    # Get the logits for the next token
                    logits, _ = model(context_tensor)
                    logits = logits[:, -1, :] / args.temperature # Pluck the last token's logits and apply temperature

                    # Apply top-k sampling
                    if args.top_k is not None:
                        v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')

                    # Calculate probabilities and sample the next token
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Stop if we generate the end-of-text token
                    if next_token.item() == enc.eot_token:
                        break
                    
                    # Print the generated token and add it to our context
                    decoded_token = enc.decode([next_token.item()])
                    print(decoded_token, end="", flush=True)

                    context_tokens.append(next_token.item())
                    context_tensor = torch.cat((context_tensor, next_token), dim=1)

        print("\n") # Newline after bot's full response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive chat with a trained GPT-2 model.")
    parser.add_argument("--checkpoint", type=str, default="log/model_optimized.pt", help="Path to the model checkpoint file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Controls randomness. Lower is more deterministic.")
    parser.add_argument("--top_k", type=int, default=200, help="Sample from the top k most likely tokens.")
    
    args = parser.parse_args()
    chat(args) 