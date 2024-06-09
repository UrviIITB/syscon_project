import cv2
import torch
import os
import numpy as np
import torchvision.transforms as transforms

model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

img_dir = '/home/ros/workspace/src/syscon_project/bot_with_camera/src/raw_imgs/'
res_dir = '/home/ros/workspace/src/syscon_project/bot_with_camera/src/annotated_imgs/'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
for img_name in os.listdir(img_dir):
    img_path = img_dir + img_name
    img = cv2.imread(img_path)
    trs_img = transform(img)
    img_tensor = trs_img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        det, da, ll = model(img_tensor)

    # DRIVEABLE AREA SEGMENTATION
    da_np = da.squeeze().detach().numpy()
    da_np = da_np[0]
    original_size = (img.shape[1], img.shape[0])
    da_resized = cv2.resize(da_np, original_size, interpolation=cv2.INTER_LINEAR)
    _, da_binary = cv2.threshold(da_resized, 0.9, 1.0, cv2.THRESH_BINARY)
    res = np.zeros_like(img)
    res[da_binary ==0] = [0, 255, 0] 
    alpha = 0.3
    annotated_image = cv2.addWeighted(img, 1, res, alpha, 0)

    path = res_dir + img_name
    cv2.imwrite(path, annotated_image)
    print("saved result {}".format(img_name))
cv2.destroyAllWindows()