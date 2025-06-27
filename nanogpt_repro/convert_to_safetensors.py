import torch
from safetensors.torch import save_model
import os
from train_gpt2 import GPT, GPTConfig

def convert_pt_to_safetensors(pt_checkpoint_path, sf_output_path):
    """
    Converts a PyTorch .pt checkpoint with tied weights to a .safetensors file.
    """
    try:
        print(f"Loading PyTorch checkpoint from: {pt_checkpoint_path}")
        # The 'weights_only=False' is required to load the custom GPTConfig object.
        # This is generally safe as we are loading a checkpoint we created ourselves.
        checkpoint = torch.load(pt_checkpoint_path, map_location="cpu")
        
        # Re-create the model from the saved configuration
        print("Re-creating model from configuration...")
        config_args = checkpoint['config']
        config = GPTConfig(**vars(config_args))
        model = GPT(config)
        
        # Load the state dictionary
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(sf_output_path), exist_ok=True)
        
        print(f"Saving model to .safetensors format at: {sf_output_path}")
        # Use save_model to correctly handle shared weights
        save_model(model, sf_output_path)
        
        print("Conversion successful!")
        print(f"Model saved to {sf_output_path}")

    except FileNotFoundError:
        print(f"Error: The file {pt_checkpoint_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the input and output file paths
    pt_path = os.path.join("log", "model_optimized.pt")
    sf_path = os.path.join("log", "model_optimized.safetensors")
    
    convert_pt_to_safetensors(pt_path, sf_path) 