import mlop
from dotenv import load_dotenv
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
import torch
from torch.cuda.amp import GradScaler, autocast
from fire import Fire 
from model import Model
from tqdm import tqdm
import torchmetrics
import random
from dataset import EmotionDataset, create_dataloaders
import wandb

load_dotenv()



def train(model, train_loader, test_loader, logger, learning_rate=1e-3, weight_decay=1e-5, epochs=10, use_amp=True, name:str = 'cnn-2', use_warmup=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if use_warmup:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=len(train_loader)//4)
    else: 
        warmup = None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=((epochs)*len(train_loader)) - (len(train_loader)//4), eta_min=0)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)
    test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)

    best_accuracy = 0
    patience = 5
    patience_counter = 0

    # check if the model is already trained and load the best model to continue training
    if (Path(os.getenv("PATH_MODEL")) / name / "best_model.pth").exists():
        model.load_state_dict(torch.load(Path(os.getenv("PATH_MODEL")) / name / "best_model.pth"))
        print("Loaded best model")
    else:
        print("No best model found, starting from scratch")

    for epoch in range(epochs):
        
        train_accuracy.reset()
        test_accuracy.reset()

        # train loop with gradient accumulation
        model.train()
        # accumulation_steps = 16  # You can adjust this value as needed
        optimizer.zero_grad()
        for i, (audio, label) in enumerate((pbar := tqdm(train_loader, desc=f"Training - Epoch {epoch+1}/{epochs}"))):
            # try:
            # audio, label, dataset_source = dataset[idx]

            audio = audio.to(device)
            label = label.to(device)
            # dataset_sources = dataset_sources.to(device)

            # Use autocast for mixed precision training
            with autocast(enabled=use_amp and device.type == 'cuda'):
                outputs = model(audio)
                loss = criterion(outputs, label)
                # loss = loss / accumulation_steps  # Normalize loss to account for accumulation

            # Backward pass with gradient scaling
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            # log the loss (un-normalized)
            logger.log({"train/loss": loss.detach().item(), "epoch": epoch, "iteration": i})

            # log the accuracy
            train_accuracy.update(outputs, label)
            logger.log({"train/accuracy": train_accuracy.compute().detach().item()})

            pbar.set_postfix({"loss": loss.item(), "accuracy": train_accuracy.compute().item()})


            if epoch == 0 and use_warmup and i <= len(train_loader)//4:
                warmup.step()
            else:
                # Step the scheduler
                scheduler.step()

            logger.log({"train/lr": optimizer.param_groups[0]['lr']})

        # test loop
        model.eval()
        # test_losses = []
        with torch.no_grad():
            for i, (audio, label) in enumerate((pbar := tqdm(test_loader, desc=f"Testing - Epoch {epoch+1}/{epochs}"))):
                audio = audio.to(device)
                label = label.to(device)
                # dataset_sources = dataset_sources.to(device)

                # Use autocast for mixed precision during inference
                with autocast(enabled=use_amp and device.type == 'cuda'):
                    outputs = model(audio)
                    loss = criterion(outputs, label)
                # test_losses.append(loss.item())
                test_accuracy.update(outputs, label)
                pbar.set_postfix({"loss": loss.item(), "accuracy": test_accuracy.compute().item(), 'iteration': i})
                logger.log({"test/loss": loss.detach().item(), "test/accuracy": test_accuracy.compute().detach().item()})
            
            # Log final metrics for the epoch
            # avg_test_loss = sum(test_losses) / len(test_losses)
            final_test_accuracy = test_accuracy.compute()
            logger.log({"avg/test/accuracy": final_test_accuracy.detach().item(), "avg/train/accuracy": train_accuracy.compute().detach().item(), "epoch": epoch})
        
        current_accuracy = test_accuracy.compute()
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            patience_counter = 0
            path_best_model = Path(os.getenv("PATH_MODEL")) / name
            path_best_model.mkdir(parents=True, exist_ok=True)
            
            # Save model state dict
            torch.save(model.state_dict(), path_best_model / "best_model.pth")
            
            # Save optimizer state for resuming training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
            }, path_best_model / "checkpoint.pth")
            
            # logger.log_artifact(path_best_model / "best_model.pth", "best_model.pth")
        else:
            patience_counter += 1
            
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_accuracy.compute():.4f}, Test Acc: {current_accuracy:.4f}, Best: {best_accuracy:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


def setup_datasets(batch_size=32):
    # load the dataset
    path_datasets = Path(os.getenv("PATH_DATASETS"))
    dataset = EmotionDataset(root_dir=path_datasets, resample_rate=16_000)

    random_indices = list(range(len(dataset)))
    random.shuffle(random_indices)
    train_indices, test_indices = train_test_split(random_indices, test_size=0.2, random_state=42)

    return dataset, train_indices, test_indices

def setup_logger(learning_rate, batch_size, weight_decay):
    print('Setting up loggers....')
    # login to mlop
    # mlop.login()

    config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
    }

    # # # 1. initialize a run
    # logger = mlop.init(
    #   project="emotion-recognition",
    #   name="cnn-1", # will be auto-generated if left unspecified
    #   config=config,
    # )
    # print('Logger setup complete')

    run = wandb.init(project="emotion-recognition", config=config)
    # run.config = config
    # run.log({"metric": 42})
    logger = run


    return logger


def main(
        learning_rate=0.02,
        batch_size=32,
        weight_decay=0,
        epochs=10,
        use_amp=True,
        use_warmup=False,
        name:str = 'cnn-2-fpr'
        ):

    train_loader, test_loader = create_dataloaders(batch_size, n_mfcc=40, n_fft=128)
    # dataset, train_indices, test_indices = setup_datasets(batch_size)
    print('Loaded datasets')
    logger = setup_logger(learning_rate, batch_size, weight_decay)
    # logger = None

    # model = Model(num_classes=8, num_mfccs_features=180)
    model = Model(num_classes=8, num_mfccs_features=40)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Log whether AMP is being used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_amp and device.type == 'cuda':
        print("Using Automatic Mixed Precision (AMP) training")
    else:
        print("Using standard precision training")

    train(model, train_loader, test_loader, logger, epochs=epochs, use_amp=use_amp, name=name, learning_rate=learning_rate)


if __name__ == "__main__":
    Fire(main)
    