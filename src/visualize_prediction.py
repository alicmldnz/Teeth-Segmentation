import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

@torch.no_grad()
def visualize_sample(model, dataset, device, num_classes, slice_index=None):
    model.eval()

    image, label = dataset[0]  # örnek bir görüntü al
    image = image.unsqueeze(0).to(device)  # [1, 1, D, H, W]

    output = model(image)  # [1, C, D, H, W]
    pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # [1, D, H, W]
    pred = pred.squeeze(0).cpu().numpy()
    label = label.numpy()

    # Hangi slice'ı göstereceğimizi belirle
    if slice_index is None:
        slice_index = pred.shape[0] // 2  # orta slice

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(pred[slice_index], cmap="nipy_spectral")
    axs[0].set_title("Tahmin")

    axs[1].imshow(label[slice_index], cmap="nipy_spectral")
    axs[1].set_title("Gerçek")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
