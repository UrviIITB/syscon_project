import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

# from YOLOP.lib.config import cfg
# from YOLOP.lib.models import get_net


# device = torch.device('cpu')
# model = get_net(cfg)
# weights = './YOLOP/weights/End-to-end.pth'
# checkpoint = torch.load(weights, map_location=device)
# model.load_state_dict(checkpoint['state_dict'])
# model = model.to(device)

model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)  # loading from repo ,loading from local device not working

im_name = 'test_img.jpg'
img_path = '/home/ros/workspace/src/syscon_project/bot_with_camera/src/' + im_name  # path to test image

transform = transforms.Compose([
    transforms.ToPILImage(),  # converting to a PIL image
    transforms.Resize((640, 640)), # resize to 640x640
    transforms.ToTensor(), # convert back to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize with specified mean and std dev
])

img = cv2.imread(img_path)
trs_img = transform(img)
img_tensor = trs_img.unsqueeze(0) # YOLOP requires batch size as well (no. of images), here just 1 so unsqueezed it to add a dimn
model.eval()
with torch.no_grad():
    det, da, ll = model(img_tensor) # object detection, area segmentation, lane detection


# DRIVEABLE AREA SEGMENTATION
da_np = da.squeeze().detach().numpy() #squeezing to remove extra dimensions and converting to numpy array
da_np = da_np[0] # removing batch dimension
original_size = (img.shape[1], img.shape[0])  # width, height
da_resized = cv2.resize(da_np, original_size, interpolation=cv2.INTER_LINEAR) # resizing to original image size
_, da_binary = cv2.threshold(da_resized, 0.5, 1.0, cv2.THRESH_BINARY)
res = np.zeros_like(img)
res[da_binary ==0] = [0, 255, 0] 
alpha = 0.3 # transparency level

# LANE SEGMENTATION, SIMILAR TO DRIVEABLE AREA SEGMENTATION
ll_np = ll.squeeze().detach().numpy()
ll_np = ll_np[0]
ll_resized = cv2.resize(ll_np, original_size, interpolation=cv2.INTER_LINEAR)
_, ll_binary = cv2.threshold(ll_resized, 0.5, 1.0, cv2.THRESH_BINARY)
res[ll_binary ==0] = [0, 0, 255]
annotated_image = cv2.addWeighted(img, 1, res, alpha, 0)  # blending the two images

# OBJECT DETECTION, GETTING MULTIPLE BOUNDING BOXES FOR SOME REASON
for detection in det[0][0]:  # detections for the first image (one image currently)
    bbox = detection[:4].numpy()  # Extracting bounding box top left coordinates and height, width
    confidence = detection[4].item()  # Extracting confidence score
    class_label = int(detection[5].item())  # Extracting class label
    
    if confidence > 0.8:
        x, y, width, height = bbox.astype(int)
        x_min = int(x - width / 2) # x_min, y_min are top left coordinates 
        y_min = int(y - height / 2)
        x_max = int(x + width / 2) # x_max , y_max are bottom right coordinates
        y_max = int(y + height / 2)
        x_min_orig = int(x_min * original_size[0] / 640) # rescaling to original image size
        y_min_orig = int(y_min * original_size[1] / 640)
        x_max_orig = int(x_max * original_size[0] / 640)
        y_max_orig = int(y_max * original_size[1] / 640)
        cv2.rectangle(annotated_image, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (255, 0, 0), 2) # drawing bounding box
        cv2.putText(annotated_image, f'Class: {class_label}, Confidence: {confidence:.2f}', (x_min_orig, y_min_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

path = '/home/ros/workspace/src/syscon_project/bot_with_camera/src/'+ 'test_img_annotated.jpg'
print(path)
cv2.imwrite(path, annotated_image)
print("saved result")
cv2.destroyAllWindows()