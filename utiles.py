import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from torchvision.utils import save_image

def save_img(input_img, output_path_, index):
    cv2.imwrite(output_path_ + "/" + "result_" + str(index) + ".jpg", input_img)
def save_tensor(input_img, output_path_, index):
    save_image(input_img, output_path_ + "/" + "result_" + str(index) + ".jpg")

def LaplacianValue(input_img):
    gray = input_img

    gaussian = cv2.GaussianBlur(gray, (3, 3), 1)
    imageVar = cv2.Laplacian(gaussian, cv2.CV_64F).var()
    
    return imageVar

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])
 
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255
 
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )
 
    return img_