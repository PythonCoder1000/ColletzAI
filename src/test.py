import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import MLP, LSTM, HybridModel, TransformerModel


def prompt_model_type():
    valid_types = ['mlp', 'lstm', 'hybrid', 'transformer']
    prompt = "Choose model type to test (mlp/lstm/hybrid/transformer) [mlp]: "
    while True:
        choice = input(prompt).strip().lower()
        if not choice:
            return 'mlp'
        if choice in valid_types:
            return choice
        print(f"Invalid choice '{choice}'. Please choose from {', '.join(valid_types)}.")


def model_default_path(model_type):
    return Path("models") / model_type / f"{model_type}_model_latest.pth"

def loadModel(model_path=None):
    if model_path is None:
        model_path = model_default_path('mlp')
    checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model_type = config.get('type', 'hybrid')
        if model_type == 'mlp':
            model = MLP(input_dim=1, hidden_dims=config['hidden_dims'], output_dim=2, dropout=config.get('dropout', 0.2))
        elif model_type == 'lstm':
            model = LSTM(input_dim=1, hidden_dim=config['hidden_dim'], num_layers=config['num_layers'], output_dim=2, dropout=config.get('dropout', 0.2), bidirectional=config.get('bidirectional', False))
        elif model_type == 'hybrid':
            model = HybridModel(input_dim=1, mlp_hidden_dims=config['mlp_hidden_dims'], lstm_hidden_dim=config['lstm_hidden_dim'], num_layers=config['num_layers'], output_dim=2, dropout=config.get('dropout', 0.2))
        elif model_type == 'transformer':
            model = TransformerModel(input_dim=1, d_model=config['d_model'], nhead=config['nhead'], num_layers=config['num_layers'], output_dim=2, dropout=config.get('dropout', 0.2))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        if 'hidden_dim' in checkpoint:
            hidden_dim = checkpoint['hidden_dim']
            num_layers = checkpoint.get('num_layers', 2)
            model = LSTM(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=2, dropout=0.2)
        elif 'hidden_dims' in checkpoint:
            hidden_dims = checkpoint['hidden_dims']
            model = MLP(input_dim=1, hidden_dims=hidden_dims, output_dim=2, dropout=0.2)
        else:
            model = HybridModel(input_dim=1, mlp_hidden_dims=[128, 256], lstm_hidden_dim=128, num_layers=2, output_dim=2, dropout=0.2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    normalization_params = {
        'input_mean': checkpoint['input_mean'],
        'input_std': checkpoint['input_std'],
        'output1_mean': checkpoint['output1_mean'],
        'output1_std': checkpoint['output1_std'],
        'output2_mean': checkpoint['output2_mean'],
        'output2_std': checkpoint['output2_std'],
    }
    
    return model, normalization_params


def predict(model, normalization_params, input_value):
    normalized_input = (input_value - normalization_params['input_mean']) / (normalization_params['input_std'] + 1e-8)
    
    input_tensor = torch.tensor([[normalized_input]], dtype=torch.float32)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    pred_steps = output[0, 0].item() * normalization_params['output1_std'] + normalization_params['output1_mean']
    pred_max = output[0, 1].item() * normalization_params['output2_std'] + normalization_params['output2_mean']
    
    return int(round(pred_steps)), int(round(pred_max))


def interactiveTest():
    model_type = prompt_model_type()
    model_path = model_default_path(model_type)
    print(f"Loading model from {model_path} ...")
    try:
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        model, norm_params = loadModel(model_path)
        print("Model loaded successfully!\n")
        if 'model_config' in checkpoint:
            print(f"Model type: {checkpoint['model_config'].get('type', 'unknown').upper()}")
            if 'best_val_loss' in checkpoint:
                print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please train the model first using src/train.py")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("="*60)
    print("Model Testing Interface")
    print("="*60)
    print("Enter input values to predict steps and max_value.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("Enter input value (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            input_value = int(user_input)
            pred_steps, pred_max = predict(model, norm_params, input_value)
            
            print(f"\nInput: {input_value}")
            print(f"Predicted Steps: {pred_steps}")
            print(f"Predicted Max Value: {pred_max}")
            print("-"*60)
            
        except ValueError:
            print("Invalid input. Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batchTest(model, norm_params, input_values):
    results = []
    for val in input_values:
        pred_steps, pred_max = predict(model, norm_params, val)
        results.append((val, pred_steps, pred_max))
    return results


if __name__ == "__main__":
    interactiveTest()
