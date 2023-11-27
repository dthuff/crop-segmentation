import matplotlib.pyplot as plt
import os


def plot_examples(X, y, y_pred, plot_path):
    """

    Parameters
    ----------
    X : torch.Tensor
    y : torch.Tensor
    y_pred : torch.Tensor
    plot_path: str

    """
    # Limit plot to 32 slices -
    if X.shape[0] > 27:
        X = X[:27, :, :, :]
        y = y[:27, :, :, :]
        y_pred = y_pred[:27, :, :, :]

    # Plot input slices X and predicted slices y_pred
    fig, axs = plt.subplots(nrows=int(X.shape[0] / 3),
                            ncols=9,
                            figsize=(9, int(X.shape[0] / 3)))
    axs = axs.flatten()

    for i, (img, gt_mask, pred_mask) in enumerate(zip(X, y, y_pred)):
        img = img.cpu().detach().squeeze().numpy()
        gt_mask = gt_mask.cpu().detach().squeeze().numpy()
        pred_mask = pred_mask.cpu().detach().squeeze().numpy()
        axs[3 * i].imshow(img.transpose(1, 2, 0), cmap='inferno', vmin=0, vmax=1)
        axs[(3 * i) + 1].imshow(gt_mask, cmap='grey', vmin=0, vmax=1)
        axs[(3 * i) + 2].imshow(pred_mask[1, :, :], cmap='grey', vmin=pred_mask[1, :, :].min(), vmax=pred_mask[1, :, :].max())

    # Hide axes and whitespace
    for a in axs:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Create the save_dir if it does not exist
    save_dir, _ = os.path.split(plot_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved example image to: {plot_path}")


def plot_and_save_loss(loss_dict, save_dir):
    """

    Args:
        loss_dict (dictionary) : dictionary containing lists of loss values per epoch
        save_dir (string) : path to save directory

    Returns:

    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    ax.plot(loss_dict["TRAIN_LOSS"], c='blue')
    ax.plot(loss_dict["VAL_LOSS"], c='cornflowerblue')

    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    fig.legend(["Loss (train)", "Loss (val)"],
               bbox_to_anchor=(0.9, 0.85))
    plt.savefig(f"{save_dir}/loss.png", dpi=150)
    plt.close(fig)
