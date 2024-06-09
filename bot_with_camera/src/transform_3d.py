import numpy as np
import cv2
import matplotlib.pyplot as plt 

# CAMERA PARAMETERS
img_dimn = 800 # in pixels,  defined in URDF file
fov = 1.3962634  # as defined in the URDF file
f = img_dimn /( 2 * np.tan(fov/2)) # = fx = fy as square pixels
cx = 400 # img_width/2
cy = 400 # img_height/2
cam_ht = 0.14 # = bot height , in URDF file
epsilon = 0.01

seg_mask = "/home/ros/workspace/src/syscon_project/bot_with_camera/src/manual/masks/2.png"

def projection(seg_mask):
    coords = []
    Z_vals = np.load("depth_map.npy")
    for u in range(seg_mask.shape[1]):
        for v in range(seg_mask.shape[0]):
            if seg_mask[u,v]==0:
                Z = Z_vals[u,v]
                if abs(Z) > 5:
                    continue
                X = (u-cx)*Z/f
                Y = -cam_ht
                coords.append([X,Y,Z])

    coords = np.array(coords)
    return coords



def plotter(coords):
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter(X,Y,Z)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.grid(True)
    ax.axis('on')
    ax.set_title("3D Projection of free space segmentation mask")
    #plt.show()
    plt.savefig("plot2.png")

def main():
    mask = cv2.imread(seg_mask, cv2.IMREAD_GRAYSCALE)
    coords = projection(mask)
    plotter(coords)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()