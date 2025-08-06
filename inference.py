import os
import hydra
import torch
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from comet_ml import Experiment

from src.datasets.asv_spoof import ASVSpoofDataset
from src.transforms.stft import STFTTransform
from src.model.lcnn_model import LCNN
from src.metrics.eer import EERMetric


@hydra.main(version_base="1.2", config_path="src/configs", config_name="model/lcnn")
def inference(cfg: DictConfig):
    """Run inference on test set and save predictions."""
    
    # Set up Comet ML
    experiment = Experiment(
        project_name=cfg.logging.comet.project,
        workspace=cfg.logging.comet.workspace,
    )
    experiment.set_name(f"{cfg.logging.comet.name}_inference")
    experiment.add_tags(cfg.logging.comet.tags + ["inference"])
    
    # Create transforms
    transform = STFTTransform(
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
        win_length=cfg.audio.win_length,
        window=cfg.audio.window,
        normalized=cfg.audio.normalized,
        center=cfg.audio.center,
        pad_mode=cfg.audio.pad_mode
    )
    
    # Create test dataset
    test_dataset = ASVSpoofDataset(
        root_dir=cfg.data.root_dir,
        partition="eval",
        transform=transform,
        sample_rate=cfg.audio.sample_rate
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.test.batch_size,
        num_workers=cfg.data.test.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    # Create model and load weights
    model = LCNN(in_channels=cfg.model.in_channels)
    model.load_state_dict(torch.load(os.path.join(cfg.save_dir, 'best_model.pth')))
    
    # Create metric
    metric = EERMetric()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Log metrics to Comet ML
    experiment.log_metrics({
        'test/eer': eer,
        'test/far': far,
        'test/frr': frr
    })
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'file_name': filenames,
        'spoof_probability': predictions
    })
    
    predictions_path = os.path.join(cfg.save_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Log predictions file to Comet ML
    experiment.log_asset(predictions_path)
    
    experiment.end()


if __name__ == '__main__':
    inference()
