import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime

model = torch.hub.load('hustvl/yolop','yolop', pretrained = True) # loading from local device not working


#taken from demo.py
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)), # resize to 640x640 pixel size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize with specified mean and std dev
])

#camera params
gazebo_cam_height = 800 # = gazebo_cam_width, defined in URDF file
fov = 1.3962634  # as defined in the URDF file
f = gazebo_cam_height /( 2 * np.tan(fov/2))
fx = f # square pixels so fx = fy = f
fy = f
cx = 400 # width/2
cy = 400 # height/2
cam_ht = 0.14 # equal to bot height, in URDF file

def pixel_to_3D(image): # converting pixel coordinates to 3D coordinates
    coords = []
    depth_map = np.load("depth_map.npy")  # calculated once in calc_depth_map.py
    for v in range(image.shape[0]): #height
        for u in range(image.shape[1]): # width
            if depth_map[v,u]!= np.inf:
                Z = depth_map[v, u]
                X = ((u - cx) * Z )/ fx
                Y = ((v - cy) * Z )/ fy
                coords.append([X, Y, Z])
    coords = np.array(coords)
    return coords

def plotter(coords): # plotting the calculated 3D coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c = coords[:, 2], cmap='viridis') # color according to depth
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Projection')
    plt_name = "./3d_plots/"+ datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".png"
    plt.savefig(plt_name)

class YOLOP_model:

    def __init__(self):
        self.image_sub = rospy.Subscriber("/turtlebot/camera1/image_raw", Image, self.callback)
        self.bridge = CvBridge() # for converting ROS to CV images
        self.image_pub = rospy.Publisher("/turtlebot/camera1/image_annotated", Image, queue_size=1)

    def callback(self,req):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req, "bgr8")
        except CvBridgeError as e:
            print(e)
        trs_image = transform(cv_image)  # for resizing and normalizing
        trs_image = trs_image.unsqueeze(0)  # batch dimension added coz 1 image only
        model.eval() # model set to evaluation mode
        with torch.no_grad():  # done when gadients already computed
            det, da, ll = model(trs_image)  # objects detected, driveable area segmentaion, lane detection

        da_np = da.squeeze().detach().numpy() # squeezing to remove extra dimensions and converting to numpy array
        da_np = da_np[0] # removing batch dimension ( as just 1 image )
        original_size = (cv_image.shape[1], cv_image.shape[0])  # width, height
        da_resized = cv2.resize(da_np, original_size, interpolation=cv2.INTER_LINEAR) # resizing to original image size
        _, da_binary = cv2.threshold(da_resized, 0.7, 1.0, cv2.THRESH_BINARY) # vals > 0.7 = 1 and rest = 0
        segmented = np.zeros_like(cv_image)
        segmented[da_binary ==0] = [0, 255, 0]  # green color for driveable area ( broadcasting color )
        alpha = 0.3 # transparency level
        annotated_img = cv2.addWeighted(cv_image, 1, segmented, alpha, 0)  # blending segmentation mask with original image
        img_name = "./annotated_images/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".jpg"
        cv2.imwrite(img_name, annotated_img)
        coords = pixel_to_3D(segmented) # converting pixel coordinates to 3D
        plotter(coords) # plotting the 3D coordinates
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_img,"bgr8"))
        except CvBridgeError as e:
            print(e)

def main():
    rospy.init_node("yolop")
    yolop = YOLOP_model()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
