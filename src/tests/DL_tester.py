import os
import sys

utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
sys.path.extend([utils_path, dataset_path])
# sys.path.append(utils_path)
# sys.path.append(dataset_path)

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from DataLoader import CelebADataset  

# Main execution
if __name__ == '__main__':
    dataset_path = r"dataset/Img/img_align_celeba"
    attr_file_path = r"dataset/Anno/list_attr_celeba.txt"
    dataset = CelebADataset(dataset_path, attr_file_path)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)  

    # Test loading a batch
    for images, labels, filenames in dataloader:
        print("Batch shape:", images.shape)  # Expected: (32, 3, 128, 128)
        print("Filenames:", filenames)  # Check filenames
        print("Labels (Blonde = 1, Not Blonde = 0):", labels.tolist())  # Check labels;
        
        break

    # Display some images with "Blond_Hair" labels and filenames
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        img, label, filename = dataset[i]
        print(f"img={filename}, label={label}")
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].axis("off")
        axes[i].set_title(f"{filename}\n{'Blonde' if label == 1 else 'Not Blonde'}")  # Show filename and label

    plt.show()