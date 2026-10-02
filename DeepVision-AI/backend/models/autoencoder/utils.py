import os
import matplotlib.pyplot as plt


def show_noisy_images(
    noisy_images,
    clean_images,
    num_images=5,
    save_path=None
):
    """
    Display clean and noisy images side-by-side.
    """

    num_images = min(num_images, len(clean_images))

    fig, axes = plt.subplots(
        2,
        num_images,
        figsize=(12, 5)
    )

    for i in range(num_images):

        clean = clean_images[i].permute(1, 2, 0).cpu().numpy()
        noisy = noisy_images[i].permute(1, 2, 0).cpu().numpy()

        axes[0, i].imshow(clean)
        axes[0, i].axis("off")

        axes[1, i].imshow(noisy)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Clean")
    axes[1, 0].set_ylabel("Noisy")

    plt.tight_layout()

    if save_path is not None:

        directory = os.path.dirname(save_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()