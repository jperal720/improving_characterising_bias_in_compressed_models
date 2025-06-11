import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),          # Convert to PyTorch tensor
])

# Custom dataset class for CelebA
class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_file):
        super().__init__()
        self.root_dir = root_dir
        self.image_files = sorted(os.listdir(root_dir))  # List all image files

        # Load annotations (load all attributes but use only "Blond_Hair" for labeling)
        self.annotations = pd.read_csv(attr_file, sep=r"\s+", header=1)
        self.annotations.replace(-1, 0, inplace=True)  # Convert -1 to 0 for binary labels

        # Ensure index is the image filename
        self.annotations.index = [f"{i:06d}.jpg" for i in range(1, len(self.annotations) + 1)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        # Get "Blond_Hair" label
        label = torch.tensor(self.annotations.loc[img_name, "Blond_Hair"], dtype=torch.float32)
        return image, label.item(), img_name  # Return image, label, and filename



