from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
import os


def open_transform(path, downsample=False):
    img = Image.open(path)
    if downsample:
        img = img.resize((96,128),Image.BICUBIC)
    img = transforms.Compose([
                            # transforms.Resize(256),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])(img)
    return img
target_img_path = "dataset/images/1VJ21D02K-Q11@8=person_half_front.jpg"
target_parse_path = "dataset/parse_cihp/1VJ21D02K-Q11@8=person_half_front.png"

source_img_path = "dataset/images/1LJ21D005-G11@10=person_half_front.jpg"
source_parse_path = "dataset/parse_cihp/1LJ21D005-G11@10=person_half_front.png"
target_img = open_transform(target_img_path, False)
source_img = open_transform(source_img_path, False)
print((np.array(Image.open(source_parse_path)) == 2))
source_parse_head = (np.array(Image.open(source_parse_path)) == 1).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 2).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 4).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 13).astype(np.float32)

target_parse_cloth = (np.array(Image.open(target_parse_path)) == 5).astype(np.float32) + \
                (np.array(Image.open(target_parse_path)) == 6).astype(np.float32) + \
                (np.array(Image.open(target_parse_path)) == 7).astype(np.float32)

phead = torch.from_numpy(source_parse_head) # [0,1]
#print(phead[30:, 98])
pcm = torch.from_numpy(target_parse_cloth) # [0,1]
im = target_img # [-1,1]
im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts --> white same as GT ...
im_h = source_img * phead - (1 - phead)

print(phead.shape)
print(pcm.shape)
print(im.shape)
print(im_c.shape)
print(im_h.shape)

def parsing_embedding(parse_path):
    parse = Image.open(parse_path) # 256,192
    parse = np.array(parse)
    parse_emb = []
    for i in range(20):
        parse_emb.append((parse == i).astype(np.float32).tolist())
    parse = np.array(parse_emb).astype(np.float32)
    return parse
#source_parse_path = os.path.join('dataset/parse_cihp', source_splitext + '.png')
source_parse = parsing_embedding(source_parse_path)

real_s = torch.tensor(source_parse)
index = [x for x in list(range(20)) if x != 5 and x != 6 and x != 7]
real_s_ = torch.index_select(real_s, 1, torch.tensor(index).cpu())
print(real_s_.shape)
# print(source_parse)
# cv2.imshow('r',(source_parse))
# cv2.waitKey(0)