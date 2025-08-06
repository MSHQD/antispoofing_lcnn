from comet_ml import Experiment
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing as mp

from src.datasets.asv_spoof import ASVSpoofDataset
from src.transforms.stft import STFTTransform
from src.model.lcnn_model import LCNN, WeightedCrossEntropyLoss
from src.metrics.eer import EERMetric


def calculate_train_eer(model, train_loader, criterion, device):
    """Calculate EER on training set."""
    model.eval()
    metric = EERMetric()
    
    with torch.no_grad():
        for batch in train_loader:
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(audio)
            metric.update(logits, labels)
    
    eer, _, _ = metric.compute()
    model.train()
    return eer


@hydra.main(version_base="1.2", config_path="src/configs", config_name="config")
def train(cfg: DictConfig):
    """Main training function."""
    
    # Debug output
    print("Config contents:")
    print(cfg)
    print("\nConfig keys:")
    print(list(cfg.keys()))
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Convert tags to list
    tags = OmegaConf.to_container(cfg.logging.comet.tags, resolve=True)
    
    # Set up Comet ML
    experiment = Experiment(
        api_key="Nvhi54f3GDZBkhMjOi8TsLHxG",
        project_name=cfg.logging.comet.project,
        workspace=cfg.logging.comet.workspace,
    )
    experiment.set_name(cfg.logging.comet.name)
    experiment.add_tags(tags)
    experiment.log_parameters(dict(cfg))
    
    # Get project root directory (2 levels up from current working directory)
    project_root = os.path.dirname(os.path.dirname(os.getcwd()))
    dataset_root = os.path.join(project_root, cfg.data.root_dir)
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    
    # Create transforms
    transform = STFTTransform(
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
        win_length=cfg.audio.win_length,
        window=cfg.audio.window,
        normalized=cfg.audio.normalized,
        center=cfg.audio.center,
        pad_mode=cfg.audio.pad_mode,
        power=cfg.audio.power
    )
    
    # Create datasets
    train_dataset = ASVSpoofDataset(
        root_dir=dataset_root,
        partition="train",
        transform=transform,
        max_audio_length=cfg.audio.max_audio_length
    )
    
    val_dataset = ASVSpoofDataset(
        root_dir=dataset_root,
        partition="dev",
        transform=transform,
        max_audio_length=cfg.audio.max_audio_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train.batch_size,
        num_workers=cfg.data.train.num_workers,
        shuffle=cfg.data.train.shuffle,
        pin_memory=False  # Set to False for MPS device
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val.batch_size,
        num_workers=cfg.data.val.num_workers,
        shuffle=cfg.data.val.shuffle,
        pin_memory=False  # Set to False for MPS device
    )
    
    # Create model
    model = LCNN(in_channels=cfg.model.in_channels)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.scheduler.T_max,
        eta_min=cfg.training.scheduler.eta_min
    )
    
    # Create criterion - paper uses standard CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create metric
    metric = EERMetric()
    
    # Training loop
    best_eer = float('inf')
    patience_counter = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    print("Starting training...")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    global_step = 0
    for epoch in range(cfg.training.epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.training.epochs}')
        for batch in progress_bar:
            # Get data
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, _ = model(audio)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.grad_clip
            )
            
            # Update weights
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            
            # Update metrics
            train_loss += loss.item() * batch_total
            train_correct += batch_correct
            train_total += batch_total
            
            # Calculate batch metrics
            batch_loss = loss.item()
            batch_accuracy = 100 * batch_correct / batch_total
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_accuracy:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log training metrics
            if global_step % cfg.logging.log_every_n_steps == 0:
                experiment.log_metrics({
                    'batch/loss': batch_loss,
                    'batch/accuracy': batch_accuracy,
                    'batch/learning_rate': scheduler.get_last_lr()[0]
                }, step=global_step)
            
            global_step += 1
        
        # Calculate epoch metrics
        train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total
        
        # Log epoch training metrics
        experiment.log_metrics({
            'train/loss': train_loss,
            'train/accuracy': train_accuracy,
            'train/epoch': epoch,
            'train/learning_rate': scheduler.get_last_lr()[0],
            'loss_train': train_loss  # Additional metric name for Comet ML
        }, step=global_step)
        
        # Validation every 3 epochs (as suggested)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            model.eval()
            metric.reset()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    # Get data
                    audio = batch['audio'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Forward pass
                    logits, _ = model(audio)
                    loss = criterion(logits, labels)
                    
                    # Update metrics
                    metric.update(logits, labels)
                    batch_total = labels.size(0)
                    val_loss += loss.item() * batch_total
                    
                    # Calculate accuracy
                    _, predicted = torch.max(logits.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += batch_total
            
            # Calculate validation metrics
            val_loss = val_loss / val_total
            val_accuracy = 100 * val_correct / val_total
            eer, far, frr = metric.compute()
            
            # Log validation metrics
            experiment.log_metrics({
                'val/loss': val_loss,
                'val/accuracy': val_accuracy,
                'val/eer': eer,
                'val/far': far,
                'val/frr': frr,
                'val/epoch': epoch,
                'loss_eval': val_loss,  # Additional metric name for Comet ML
                'eer_eval': eer  # Additional metric name for Comet ML
            }, step=global_step)
            
            # Calculate train EER
            train_eer = calculate_train_eer(model, train_loader, criterion, device)
            
            # Log combined metrics
            experiment.log_metrics({
                'loss_train': train_loss,
                'loss_eval': val_loss,
                'eer_train': train_eer,
                'eer_eval': eer
            }, step=global_step)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            print(f"Val EER: {eer:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}")
            print("-" * 50)
            
            # Early stopping
            if eer < best_eer - cfg.training.early_stopping.min_delta:
                best_eer = eer
                patience_counter = 0
                
                # Save best model
                model_path = os.path.join(cfg.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_eer': best_eer,
                    'config': dict(cfg)
                }, model_path)
                
                # Log best model to Comet ML
                experiment.log_model('best_model', model_path)
                experiment.log_metrics({
                    'best/eer': best_eer,
                    'best/epoch': epoch
                }, step=global_step)
            else:
                patience_counter += 1
                
            if patience_counter >= cfg.training.early_stopping.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Update scheduler
        scheduler.step()
    
    experiment.end()


if __name__ == '__main__':
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn')
    train()
