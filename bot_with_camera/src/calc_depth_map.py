import numpy as np
import cv2


gazebo_cam_height = 800 # = gazebo_cam_width, defined in URDF file
fov = 1.3962634  # as defined in the URDF file
f = gazebo_cam_height /( 2 * np.tan(fov/2))
fx = f # ( as square pixels)
fy = f # ( as square pixels)
cx = 400 # width/2
cy = 400 # height/2
cam_ht = 0.14 # = bot height , in URDF file
epsilon = 0.00000001  # 10^-8 


# ASSUMING FIXED CAMERA HEIGHT WITH 0 PITCH ANGLE

# if camera has 0 pitch angle, depth of any point is same as the point 
# vertically below it on the ground, also tan(A) = h/z = y/z = known 
# here h is the camera height and z is the depth of the point
# so we get z = h/tan(A) = h/ (y/z)
def calc_depth_map(image):
    height, width = image.shape[0], image.shape[1]
    depth_map = np.zeros((height, width))
    for i in range(height):
        y_z = (i - cy) / fy
        if np.abs(y_z) >= epsilon:
            Z = cam_ht / np.abs(y_z)
        else:
            Z = np.inf
        depth_map[i, :] = Z
    return depth_map


if __name__=="__main__":
    image = cv2.imread('sample.jpg')
    depth_map = calc_depth_map(image)
    np.save('depth_map.npy', depth_map)
    print("saved depth map")
    cv2.destroyAllWindows()