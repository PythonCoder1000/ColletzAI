import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import CollatzDataset, MLP, LSTM, HybridModel, TransformerModel, EarlyStopping


def mean_loss(total_loss, num_batches, stage):
    if num_batches == 0:
        raise RuntimeError(f"No batches processed during {stage}. Adjust your dataset split or batch size.")
    return total_loss / num_batches



def resolve_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def trainEpoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return mean_loss(total_loss, num_batches, "training")


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return mean_loss(total_loss, num_batches, "validation")


def trainModel(model, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10, 
               device='cuda', optimizer_type='adam', scheduler_type='reduce_on_plateau',
               weight_decay=1e-5, max_grad_norm=1.0, warmup_epochs=0):
    criterion = torch.nn.MSELoss()
    
    optimizer_kind = optimizer_type.lower()
    if optimizer_kind == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_kind == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_kind == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler_kind = scheduler_type.lower()
    if scheduler_kind == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif scheduler_kind == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_kind == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_kind == 'warmup_cosine':
        effective_warmup = max(1, warmup_epochs)
        decay_epochs = max(1, num_epochs - effective_warmup)
        def lr_lambda(epoch):
            if epoch < effective_warmup:
                return float(epoch + 1) / float(effective_warmup)
            progress = min(max((epoch - effective_warmup) / decay_epochs, 0.0), 1.0)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    early_stopping = EarlyStopping(patience=patience)
    
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        train_loss = trainEpoch(model, train_loader, criterion, optimizer, device, max_grad_norm)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if scheduler is not None:
            if scheduler_kind == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return train_losses, val_losses, learning_rates


def testModel(model, test_loader, dataset, device='cuda'):
    model.eval()
    predictions = []
    actuals = []
    inputs_list = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            pred_steps = outputs[:, 0].cpu().numpy() * dataset.output1_std + dataset.output1_mean
            pred_max = outputs[:, 1].cpu().numpy() * dataset.output2_std + dataset.output2_mean
            
            actual_steps = targets[:, 0].cpu().numpy() * dataset.output1_std + dataset.output1_mean
            actual_max = targets[:, 1].cpu().numpy() * dataset.output2_std + dataset.output2_mean
            
            denorm_inputs = (inputs.cpu().numpy() * dataset.input_std + dataset.input_mean).squeeze()
            
            for i in range(len(pred_steps)):
                if denorm_inputs.ndim == 1:
                    inputs_list.append(denorm_inputs[i])
                else:
                    inputs_list.append(denorm_inputs[i][0])
                predictions.append((pred_steps[i], pred_max[i]))
                actuals.append((actual_steps[i], actual_max[i]))
    
    return inputs_list, predictions, actuals


def printTestResults(inputs, predictions, actuals, num_samples=10):
    if not predictions:
        print("\nNo test samples available.")
        return
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"{'Input':<10} {'Pred Steps':<15} {'Actual Steps':<15} {'Pred Max':<15} {'Actual Max':<15} {'Steps Error':<15} {'Max Error':<15}")
    print("-"*80)
    
    for i in range(min(num_samples, len(inputs))):
        input_val = int(inputs[i])
        pred_steps, pred_max = predictions[i]
        actual_steps, actual_max = actuals[i]
        steps_error = abs(pred_steps - actual_steps)
        max_error = abs(pred_max - actual_max)
        
        print(f"{input_val:<10} {pred_steps:<15.2f} {actual_steps:<15.2f} {pred_max:<15.2f} {actual_max:<15.2f} {steps_error:<15.2f} {max_error:<15.2f}")
    
    steps_errors = [abs(p[0] - a[0]) for p, a in zip(predictions, actuals)]
    max_errors = [abs(p[1] - a[1]) for p, a in zip(predictions, actuals)]
    
    print("\n" + "-"*80)
    print(f"Mean Steps Error: {np.mean(steps_errors):.2f}")
    print(f"Mean Max Error: {np.mean(max_errors):.2f}")
    print(f"Median Steps Error: {np.median(steps_errors):.2f}")
    print(f"Median Max Error: {np.median(max_errors):.2f}")
    print("="*80)



def main():
    data_file = "data/data.txt"
    batch_size = 64
    num_epochs = 150
    learning_rate = 0.001
    patience = 20
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    dropout = 0.2
    model_type = 'mlp'
    mlp_hidden_dims = [256, 512, 256, 128]
    lstm_hidden_dim = 128
    num_layers = 2
    transformer_d_model = 128
    transformer_nhead = 4
    transformer_num_layers = 3
    optimizer_type = 'adamw'
    scheduler_type = 'reduce_on_plateau'
    weight_decay = 1e-5
    max_grad_norm = 1.0
    warmup_epochs = 5
    
    device = resolve_device()
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    full_dataset = CollatzDataset(data_file)
    total_samples = len(full_dataset)
    if total_samples == 0:
        raise ValueError("Dataset is empty.")
    print(f"Total samples: {total_samples}")
    
    split_array = np.array([train_split, val_split, test_split], dtype=np.float64)
    if np.any(split_array < 0):
        raise ValueError("Split ratios must be non-negative.")
    split_sum = split_array.sum()
    if split_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    split_array /= split_sum
    lengths = (split_array * total_samples).astype(int)
    remainder = total_samples - lengths.sum()
    for i in range(remainder):
        lengths[i % len(lengths)] += 1
    train_size, val_size, test_size = lengths.tolist()
    if train_size == 0 or val_size == 0:
        raise ValueError("Not enough data to form non-empty train and validation sets with the provided splits.")
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=generator
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    
    print(f"\nCreating {model_type.upper()} model...")
    if model_type == 'mlp':
        model = MLP(input_dim=1, hidden_dims=mlp_hidden_dims, output_dim=2, dropout=dropout)
        model_config = {'type': 'mlp', 'hidden_dims': mlp_hidden_dims, 'dropout': dropout}
    elif model_type == 'lstm':
        model = LSTM(input_dim=1, hidden_dim=lstm_hidden_dim, num_layers=num_layers, 
                     output_dim=2, dropout=dropout, bidirectional=True)
        model_config = {'type': 'lstm', 'hidden_dim': lstm_hidden_dim, 'num_layers': num_layers, 
                       'dropout': dropout, 'bidirectional': True}
    elif model_type == 'hybrid':
        model = HybridModel(input_dim=1, mlp_hidden_dims=[128, 256], lstm_hidden_dim=lstm_hidden_dim,
                           num_layers=num_layers, output_dim=2, dropout=dropout)
        model_config = {'type': 'hybrid', 'mlp_hidden_dims': [128, 256], 'lstm_hidden_dim': lstm_hidden_dim,
                       'num_layers': num_layers, 'dropout': dropout}
    elif model_type == 'transformer':
        model = TransformerModel(input_dim=1, d_model=transformer_d_model, nhead=transformer_nhead,
                                num_layers=transformer_num_layers, output_dim=2, dropout=dropout)
        model_config = {'type': 'transformer', 'd_model': transformer_d_model, 'nhead': transformer_nhead,
                       'num_layers': transformer_num_layers, 'dropout': dropout}
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    print(f"\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"\nTraining configuration:")
    print(f"  Optimizer: {optimizer_type}")
    print(f"  Scheduler: {scheduler_type}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Max grad norm: {max_grad_norm}")
    
    print("\nStarting training...")
    train_losses, val_losses, learning_rates = trainModel(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        patience=patience,
        device=device,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        warmup_epochs=warmup_epochs
    )
    
    print("\nTesting model...")
    inputs, predictions, actuals = testModel(model, test_loader, full_dataset, device)
    printTestResults(inputs, predictions, actuals, num_samples=20)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/{model_type}_model_{timestamp}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    latest_path = f"models/{model_type}_model_latest.pth"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'input_mean': full_dataset.input_mean,
        'input_std': full_dataset.input_std,
        'output1_mean': full_dataset.output1_mean,
        'output1_std': full_dataset.output1_std,
        'output2_mean': full_dataset.output2_mean,
        'output2_std': full_dataset.output2_std,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'optimizer_type': optimizer_type,
        'scheduler_type': scheduler_type,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'best_val_loss': min(val_losses) if val_losses else None,
    }
    
    torch.save(checkpoint, model_path)
    torch.save(checkpoint, latest_path)
    
    history_path = f"models/{model_type}_history_{timestamp}.json"
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'model_config': model_config,
        'training_config': {
            'optimizer': optimizer_type,
            'scheduler': scheduler_type,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
        }
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nModel saved to {model_path}")
    print(f"Latest model saved to {latest_path}")
    print(f"Training history saved to {history_path}")
    if val_losses:
        print(f"\nBest validation loss: {min(val_losses):.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
    else:
        print("\nValidation losses unavailable.")


if __name__ == "__main__":
    main()
