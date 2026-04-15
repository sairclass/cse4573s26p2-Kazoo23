'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import kornia.geometry.transform
import torch
import kornia as K
import kornia.feature as KF
from typing import Dict

from kornia.geometry import RANSAC

from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #

def extract_features(img):
    sift = KF.SIFTFeature(num_features=2048)
    imgGray = kornia.color.rgb_to_grayscale(img)
    keypoints,score,descriptors = sift(imgGray)
    return keypoints.squeeze(0),descriptors.squeeze(0)


def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    keys = list(imgs.keys())

    img1 = imgs[keys[0]].float() / 255.0

    img2 = imgs[keys[1]].float() / 255.0

    points1,desc1 = extract_features(img1.unsqueeze(0))
    points2,desc2 = extract_features(img2.unsqueeze(0))

    dist,indexes = KF.match_smnn(desc1, desc2, 0.8)

    if len(indexes) < 4:
        raise RuntimeError('Not enough features to stitch background.')

    points1 = points1[indexes[:,0]][:,:,2]

    points2 = points2[indexes[:,1]][:,:,2]

    ransac = RANSAC(model_type="homography", inl_th=5, confidence=0.99, max_iter=1000)

    shape, mask = ransac(points1,points2)

    inliers = mask.sum().item()

    if inliers < 10:
        raise RuntimeError('Not enough inliers to stitch background.')

    h1, w1 = img1.shape[1], img1.shape[2]
    h2, w2 = img2.shape[1], img2.shape[2]

    H1 = shape

    corners1 = torch.tensor([[0,0],[w1-1,0],[0,h1-1],[w1-1,h1-1]],dtype=torch.float32)

    corners2 = torch.tensor([[0, 0], [w2 - 1, 0], [0, h2 - 1], [w2 - 1, h2 - 1]], dtype=torch.float32)

    temp = torch.ones((4,1),dtype=torch.float32)
    corner2H = torch.cat([corners2,temp],dim=1) @ H1.T
    corners2proj = corner2H[:,:2] / corner2H[:,2:3]

    canvasCorners = torch.cat([corners1, corners2proj], dim=0)

    minxy = canvasCorners.min(dim=0)[0]
    maxxy = canvasCorners.max(dim=0)[0]

    T = torch.eye(3, dtype=torch.float32)
    T[0,2] = -minxy[0]
    T[1,2] = -minxy[1]

    h1 = T.clone()

    h2 = torch.mm(T,shape)

    width = int(maxxy[0] - minxy[0]+1)
    height = int(maxxy[1] - minxy[1]+1)

    img1warped = K.geometry.warp_perspective(img1.unsqueeze(0),h2.unsqueeze(0),dsize=(height,width), mode='nearest',padding_mode='fill')[0]

    img2warped = K.geometry.warp_perspective(img2.unsqueeze(0),h1.unsqueeze(0),dsize=(height,width), mode='nearest',padding_mode='fill')[0]

    canvas = torch.zeros(3,height,width)

    img1mask = (img1warped > 0)

    img2mask = (img2warped> 0)

    canvas[img1mask & ~img2mask] = img1warped[img1mask & ~img2mask]

    canvas[img2mask & ~img1mask] = img2warped[img2mask & ~img1mask]

    canvas[img1mask & img2mask] = (img1warped[img1mask & img2mask] + img2warped[img2mask & img1mask ]) / 2

    blackmask = (canvas > 0).any(dim=0)

    fixedwidth = blackmask.any(dim=0).nonzero()

    canvas = canvas[:, :, fixedwidth[0].item():fixedwidth[-1].item() + 1]

    img = (canvas * 255).to(torch.uint8)

    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama,
        overlap: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    keys = list(imgs.keys())

    ransac = RANSAC(model_type="homography", inl_th=5, confidence=0.99, max_iter=1000)

    length = len(keys)

    overlap = torch.zeros(length,length)

    for i in range(length):
        overlap[i][i] = 1

    for i in range(0,length):
        img1 = imgs[keys[i]].float() / 255.0
        temppoints1, tempdesc1 = extract_features(img1.unsqueeze(0))
        for j in range(i+1,length):
            img2 = imgs[keys[j]].float() / 255.0
            temppoints2, tempdesc2 = extract_features(img2.unsqueeze(0))

            dist, indexes = KF.match_smnn(tempdesc1, tempdesc2, 0.8)

            if len(indexes) > 4:
                points1 = temppoints1[indexes[:, 0]][:, :, 2]

                points2 = temppoints2[indexes[:, 1]][:, :, 2]

                shape, mask = ransac(points1, points2)

                inliers = mask.sum().item()

                if inliers > 10:
                    overlap[i][j] = 1
                    overlap[j][i] = 1

    anchorindex = overlap.sum(dim=1).argmax().item()

    anchorimage = imgs[keys[anchorindex]]

    chosenimages = [keys[anchorindex]]

    while len(chosenimages) < length:
        for i in range(length):
            if i == anchorindex:
                continue
            currimgs = {
                keys[anchorindex]: anchorimage,
                keys[i]: imgs[keys[i]],
            }
            try:
                anchorimage = stitch_background(currimgs)
            except RuntimeError as e:
                chosenimages.append(keys[i])
                continue

            blackmask = (anchorimage > 0).any(dim=0)

            fixedwidth = blackmask.any(dim=0).nonzero()

            fixedheight = blackmask.any(dim=1).nonzero()

            anchorimage = anchorimage[:,fixedheight[0].item():fixedheight[-1].item()+1,fixedwidth[0].item():fixedwidth[-1].item()+1]

            chosenimages.append(keys[i])
    overlap = overlap.to(torch.uint8)

    img = anchorimage

    return img, overlap
