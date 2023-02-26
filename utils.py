import cv2
import torch
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt


NUM_WORKER = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# consine similarity
def cosine(query_feat, gallery_feat):
    return np.dot(query_feat, gallery_feat) / (norm(query_feat) * norm(gallery_feat))


# euclidean distance
def euclidean(query_feat, gallery_feat):
    return norm(query_feat - gallery_feat)


def visualization(retrieved, query):
    plt.subplot(2, 3, 1)
    plt.title("query")
    query_img = cv2.imread(query)
    img_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb)
    for i in range(5):
        img_path = './data/gallery/' + retrieved[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        plt.subplot(2, 3, i+2)
        plt.title(retrieved[i][1])
        plt.imshow(img_rgb)
    plt.show()


def save_rank(filename, data, ord="asc"):
    # data: 28,493 * 50
    with open(filename, "w+") as f:
        for query_idx in range(len(data[0])):
            sub_data = np.array(data[:][query_idx])
            res = np.argsort(sub_data) if ord == "asc" else np.argsort(-sub_data)
            res_str = f"Q{query_idx+1}: " + " ".join([str(num) for num in res]) + "\n"
            f.write(res_str)


if __name__ == "__main__":
    a = np.array([1,1,1])
    b = np.array([2,2,2])
    sim = cosine(a, b)
    print(sim)