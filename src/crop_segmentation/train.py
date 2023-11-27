import torch
import yaml
from crop_segmentation.plotting import plot_examples


scaler = torch.cuda.amp.GradScaler()


def parse_training_config(config_path: str):
    with open(config_path, "r") as stream:
        try:
            ll = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return ll


def train_loop(dataloader, model, loss_fn, optimizer, epoch_number, save_dir: str):
    """TRAIN_LOOP - Runs training for one epoch

    Args:
        dataloader (torch.DataLoader): dataloader for training set
        model (nn.Module): model object
        loss_fn (nn.Module): loss
        optimizer (torch.Optimizer): model optimizer

    Returns:
        mean_loss_kl
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):

        # Send the inputs X and labels y to the GPU
        X = X.cuda()
        y = y.cuda()

        # One hot encode labels
        y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=2)
        y_one_hot = y_one_hot.squeeze().permute(0, 3, 1, 2)

        # Compute prediction and loss
        y_pred = model(X)
        batch_loss = loss_fn(y_one_hot, y_pred)

        # Backpropagation
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Add this batch loss to total loss
        total_loss += batch_loss.item()

        # Print loss every 10 batches
        if batch % 10 == 0:
            loss, current = batch_loss.item(), batch * len(X)
            print(f"Total loss: {loss:.2f}  [{current:>2d}/{size:>2d}]")

        # Plot a montage of X and y_pred comparisons for the first batch
        if batch == 0:
            plot_examples(X=X.cpu(), y=y.cpu(), y_pred=y_pred.cpu(),
                          plot_path=f"{save_dir}/train_epoch_{str(epoch_number)}.png")

    # Return the mean per-batch loss
    return total_loss / num_batches


def val_loop(dataloader, model, loss_fn, epoch_number, save_dir: str):
    """VAL_LOOP - Runs validation for one epoch

    Args:
        dataloader (torch.DataLoader): dataloader for validation set
        model (nn.Module): model object
        loss_fn (nn.Module): loss
        epoch_number (int): epoch counter for saving plots
    """
    # Stop training during validation
    model.eval()

    num_batches = len(dataloader)
    loss = 0
    plotted_this_epoch = False

    with torch.no_grad():
        for X, y in dataloader:
            # Send the inputs X and labels y to the GPU
            X = X.cuda()
            y = y.cuda()

            # One hot encode labels
            y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=2)
            y_one_hot = y_one_hot.squeeze().permute(0, 3, 1, 2)

            # Compute prediction and loss
            y_pred = model(X)
            loss += loss_fn(y_one_hot, y_pred).item()

            # Plot a montage of X and y_pred comparisons for the first batch
            if epoch_number % 2 == 0 and not plotted_this_epoch:
                plot_examples(X=X.cpu(), y=y.cpu(), y_pred=y_pred.cpu(),
                              plot_path=f"{save_dir}/epoch_{str(epoch_number)}.png")
                plotted_this_epoch = True

    return loss / num_batches
