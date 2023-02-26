import tqdm
import argparse
from feature import vgg11_feat
from utils import NUM_WORKER, cosine, save_rank
from dataset import MyDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


if __name__ == "__main__":
    # load data
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    gallery_dataset = MyDataset("data/gallery", transform=transform)
    query_dataset = MyDataset("data/query", transform=transform)

    batch_size = 64
    gallery_size = len(gallery_dataset)
    query_size = len(query_dataset)

    gallery_dataloader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER)
    query_dataloader = DataLoader(query_dataset, batch_size=query_size, shuffle=False, num_workers=NUM_WORKER)

    # pure VGG11
    for data in query_dataloader:
        query_features = vgg11_feat(data)  # (50, 512, 7, 7)
        query_features = query_features.reshape(query_size, -1)
        break

    cosine_sim = []
    # euclidean_dists = []
    for data in tqdm.tqdm(gallery_dataloader):
        features = vgg11_feat(data)  # (64, 512, 7, 7)
        features = features.reshape(batch_size, -1)
        cosine_sim += [[cosine(query, gallery) for query in query_features] for gallery in features]  # 64*50
    save_rank(filename="rankList.txt", data=cosine_sim, ord="desc")
