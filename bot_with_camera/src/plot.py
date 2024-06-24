import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import cv2
import transformation

name = "1"

seg_mask = "/home/ros/workspace/src/syscon_project/bot_with_camera/src/manual/masks/{}.png".format(name)
plot_path = "/home/ros/workspace/src/syscon_project/bot_with_camera/src/manual/plots/"

mask = cv2.imread(seg_mask, cv2.IMREAD_GRAYSCALE)
points = []
for u in range(mask.shape[1]):
    for v in range(450,mask.shape[0]):
        if mask[v,u]==0:
            coord = transformation.pixel_to_world(np.array([u,v,1]))
            points.append(coord)

x = np.array([coord[0] for coord in points])
y = np.array([coord[1] for coord in points])
z = [0 for coord in points]

# #3D PLOT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-1.2,1.2)
ax.set_ylim(0.1,1.4)
ax.set_zlim(-0.4,0.4)
plt.savefig(plot_path+"3d_plots/" + "{}.png".format(name))

#2D PLOT (TOP VIEW SHOWING DISTANCE FROM BOT)
distances = np.sqrt(x**2 + y**2)
norm_distances = (distances - distances.min()) / (distances.max() - distances.min())
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=norm_distances, cmap='viridis', marker='o')
colorbar = plt.colorbar(scatter)
colorbar.set_label('Normalized Distance from bot')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Plot')
plt.xlim(-1.2,1.2)
plt.ylim(0.1,1.4)
plt.savefig(plot_path+"2d_plots/" + "{}.png".format(name))
