import numpy as np
import cv2

# CAMERA PARAMETERS
img_dimn = 800 # in pixels,  defined in URDF file
fov = 1.3962634  # as defined in the URDF file
f = img_dimn /( 2 * np.tan(fov/2)) # = fx = fy as square pixels  ~ 479.7
cx = 400 # img_width/2
cy = 400 # img_height/2
cam_ht = 0.14 # = bot height , in URDF file
epsilon = 0.1

img_path = '/home/ros/workspace/src/syscon_project/bot_with_camera/src/sample_img.jpg'


# If pixel belongs to free space , it will have corresponding y coordinate = -h 
# where h is the height of the camera from the ground = robot height
# Also , Y = Z * (v-Cy)/f from this calculate Z = -h * f/(v-Cy), here (Cx, Cy) is principal point

def depth_map(image):
    Z = np.zeros((image.shape[0], image.shape[1]))
    for v in range(image.shape[0]):
        if np.abs(v - cy)>=epsilon:
            Z[:,v] = -1*cam_ht * f/(v-cy)
        else:
           Z[:,v] = -1*cam_ht * f/epsilon

    np.save("depth_map.npy", Z)

def main():
    img = cv2.imread(img_path)
    depth_map(img)
    print("saved depth map")
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()