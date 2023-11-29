import torch
import yaml
from crop_segmentation.plotting import plot_examples


def parse_training_config(config_path: str) -> dict:
    """
    Parse the config and return as dict
    Parameters
    ----------
    config_path: str
        Path to config yml

    Returns
    -------
    Nested dict containing config fields.
    """
    with open(config_path, "r") as cfg:
        try:
            ll = yaml.safe_load(cfg)
        except yaml.YAMLError as exc:
            print(exc)
    return ll


def train_loop(dataloader, model, loss_fn, optimizer, epoch_number, config: dict) -> float:
    """
    Runs training for one epoch.

    Parameters
    ----------
    dataloader: torch.DataLoader
        Dataloader for training dataset
    model: nn.Module
        Model object
    loss_fn: nn.Module
        Loss object
    optimizer: torch.Optimizer
        Model optimizer
    epoch_number: int
        Epoch number
    config: dict
        Config dictionary object

    Returns
    -------
    mean_per_batch_loss: float
        Loss averaged over batches
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.cuda()
        y = y.cuda()

        y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=2)
        y_one_hot = y_one_hot.squeeze().permute(0, 3, 1, 2)

        y_pred = model(X)
        batch_loss = loss_fn(y_one_hot, y_pred)

        # Backpropagation
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item()

        if batch % 10 == 0:
            loss, current = batch_loss.item(), batch * len(X)
            print(f"Total loss: {loss:.2f}  [{current:>2d}/{size:>2d}]")

        # Plot a montage of X and y_pred comparisons for the first batch
        if epoch_number % config['logging']['plot_every_n_epochs'] == 0 and batch == 0:
            plot_examples(X=X.cpu(), y=y.cpu(), y_pred=y_pred.cpu(),
                          plot_path=f"{config['logging']['plot_save_dir']}/train_epoch_{str(epoch_number)}.png")

    mean_per_batch_loss = total_loss / num_batches
    return mean_per_batch_loss


def val_loop(dataloader, model, loss_fn, epoch_number, config: dict):
    """
    Run validation for one epoch.

    Parameters
    ----------
    dataloader: torch.DataLoader
        Dataloader for training dataset
    model: nn.Module
        Model object
    loss_fn: nn.Module
        Loss object
    epoch_number: int
        Epoch number
    config: dict
        Config dictionary object

    Returns
    -------
    mean_per_batch_loss: float
        Mean loss over val set.
    """
    # Stop training during validation
    model.eval()

    num_batches = len(dataloader)
    loss = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()

            y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=2)
            y_one_hot = y_one_hot.squeeze().permute(0, 3, 1, 2)

            y_pred = model(X)
            loss += loss_fn(y_one_hot, y_pred).item()

            # Plot a montage of X and y_pred comparisons for the first batch
            if epoch_number % config['logging']['plot_every_n_epochs'] == 0 and batch == 0:
                plot_examples(X=X.cpu(), y=y.cpu(), y_pred=y_pred.cpu(),
                              plot_path=f"{config['logging']['plot_save_dir']}/val_epoch_{str(epoch_number)}.png")

    mean_per_batch_loss = loss / num_batches
    return mean_per_batch_loss
