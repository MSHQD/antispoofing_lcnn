import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.asv_spoof import ASVSpoofDataset
from src.transforms.stft import STFTTransform
from src.model.lcnn_model import LCNN
from src.metrics.eer import EERMetric


def inference():
    """Run inference on test set and save predictions."""
    
    print("Starting inference...")
    
    # Configuration
    config = {
        'audio': {
            'n_fft': 512,
            'hop_length': 160,
            'win_length': 400,
            'window': 'hann',
            'normalized': True,
            'center': True,
            'pad_mode': 'reflect',
            'sample_rate': 16000
        },
        'data': {
            'root_dir': 'src/datasets/ASVspoof2019',
            'test': {
                'batch_size': 32,
                'num_workers': 4
            }
        },
        'model': {
            'in_channels': 1
        },
        'save_dir': 'experiments/lcnn_asv'
    }
    
    # Create transforms
    transform = STFTTransform(
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        win_length=config['audio']['win_length'],
        window=config['audio']['window'],
        normalized=config['audio']['normalized'],
        center=config['audio']['center'],
        pad_mode=config['audio']['pad_mode']
    )
    
    # Create test dataset
    test_dataset = ASVSpoofDataset(
        root_dir=config['data']['root_dir'],
        partition="eval",
        transform=transform,
        sample_rate=config['audio']['sample_rate']
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['test']['batch_size'],
        num_workers=config['data']['test']['num_workers'],
        shuffle=False,
        pin_memory=True
    )
    
    # Create model and load weights
    model = LCNN(in_channels=config['model']['in_channels'])
    
    # Check if model file exists
    model_path = os.path.join(config['save_dir'], 'best_model.pth')
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    # Create metric
    metric = EERMetric()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Inference
    model.eval()
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # Get data
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            print(f"Audio tensor shape: {audio.shape}")
            print(f"Labels tensor shape: {labels.shape}")
            
            # Forward pass
            logits, _ = model(audio)
            
            # Update metric
            metric.update(logits, labels)
            
            # Store predictions and filenames
            probs = torch.softmax(logits, dim=1)
            spoof_probs = probs[:, 1].cpu().numpy()
            predictions.extend(spoof_probs)
            filenames.extend(batch['file_name'])
    
    # Compute final metrics
    eer, far, frr = metric.compute()
    print(f"Test set metrics:")
    print(f"EER: {eer:.4f}")
    print(f"FAR: {far:.4f}")
    print(f"FRR: {frr:.4f}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'file_name': filenames,
        'spoof_probability': predictions
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(config['save_dir'], exist_ok=True)
    
    predictions_path = os.path.join(config['save_dir'], 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


if __name__ == '__main__':
    inference()