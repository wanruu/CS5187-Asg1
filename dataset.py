import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class MyDataset(Dataset):
    def __init__(self, dirname, transform=None):
        self.dirname = dirname
        self.transform = transform
        self.img_names = os.listdir(dirname)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        path = os.path.join(self.dirname, self.img_names[idx])
        image = read_image(path) / 255.0
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == "__main__":
    gallery = MyDataset("data/gallery")
    query = MyDataset("data/query")

    for image in gallery:
        print(image.shape)
        break
    
    