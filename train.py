import torch

scaler = torch.cuda.amp.GradScaler()


def train_loop(dataloader, model, loss_fn, optimizer, amp_on):
    """TRAIN_LOOP - Runs training for one epoch

    Args:
        dataloader (torch.DataLoader): dataloader for training set
        model (nn.Module): model object
        loss_fn (nn.Module): loss
        optimizer (torch.Optimizer): model optimizer
        amp_on (Boolean): enable automatic mixed precision

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

        # Compute prediction and loss
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
            y_pred = model(X)
            batch_loss = loss_fn(y, y_pred)

        # Backpropagation - with GradScaler for optional automatic mixed precision
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Add this batch loss to total loss
        total_loss += batch_loss.item()

        # Print loss every 10 batches
        if batch % 10 == 0:
            loss, current = batch_loss.item(), batch * len(X)
            print(f"Total loss: {loss:.2f}  [{current:>2d}/{size:>2d}]")

    # Return the mean per-batch loss
    return total_loss / num_batches


def val_loop(dataloader, model, loss_fn, epoch_number):
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

            # Compute prediction and loss
            y_pred = model(X)
            loss += loss_fn(y, y_pred).item()

            """
            # Plot a montage of X and y_pred comparisons for the first batch
            if epoch_number % 10 == 0 and not plotted_this_epoch:
                plot_examples(X=X.cpu(),
                              y_pred=y_pred.cpu(),
                              plot_path="./saved_models/validation_images/epoch_" + str(epoch_number) + ".png")
                plotted_this_epoch = True
            """

    return loss / num_batches