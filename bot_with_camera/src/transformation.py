'''
WORLD COORDINATE FRAME          ---------> u  PIXEL FRAME
z                               |
^    y                          | 
|   /                           |
|  /                            v
| /
--------> x

CAMERA COORDINATE FRAME
   z
  /
 /
/
--------> x
|
|
|
v
y

rotation matrix R = [1 0  0]
                    [0 0 -1]
                    [0 1  0]

translation matrix T = [ 0 ]
                       [ h ]
                       [ 0 ]

In world frame, free space points have z=0

i.e world frame coordinates = [x_w, y_w, 0].T

[ x_c ] = [1 0  0] * [x_w] + [ 0 ]
[ y_c ]   [0 0 -1]   [y_w]   [ h ]
[ z_c ]   [0 1  0]   [ 0 ]   [ 0 ]

=> x_c = x_w
=> y_c = h
=> z_c = y_w

[ x_c ]    [ x_c/z_c ]     [  x_w/y_w]
[ y_c ] => [ y_c/z_c ]  =  [   h/y_w ]
[ z_c ]    [    1    ]     [     1   ]

Calibration matrix (intrinsic matrix) k = [fx 0 cx]
                                          [0 fy cy]
                                          [0  0  1]

[fx 0 cx] * [ x_c/z_c ] = [u]
[0 fy cy] * [ y_c/z_c ]   [v]
[0  0  1] * [    1    ]   [1]

i.e.,

[fx 0 cx] * [  x_w/y_w] = [u]
[0 fy cy] * [   h/y_w ]   [v]
[0  0  1] * [    1    ]   [1]

=> u = fx * x_w/y_w + cx
=> v = fy * h/y_w + cy

=> y_w = h * fy/(v-cy)
=> x_w = (u-cx) * y_w/fx = [ (u-cx)*h*fy ]/ [ (v-cy)*fx ] = [(u-cx)*h]/(v-cy)


# CAMERA PARAMETERS
img_dimn = 800 pixels
fov = 1.3962634
f = img_dimn /( 2 * np.tan(fov/2)) = 476.70143780997665 pixels
cx = 400  (img_width/2 )
cy = 400  (img_height/2 )
cam_ht = 0.14 = bot_height = h

this gives y_w = 66.73820129339674 / (v - 400)

and x_w = (u-400) * 0.14/ (v-400)

'''

import numpy as np
f = 476.70143780997665
cx = 400
cy = 400
R = np.array([[1,0,0],[0,0,-1],[0,1,0]])
T = np.array([[0],[0.14],[0]])
K = np.array([[f,0,cx],[0,f,cy],[0,0,1]])


def world_to_pixel(w):
    c = np.matmul(R,w) + T
    c_n = c/c[2]
    p = np.matmul(K,c_n)
    return p

def pixel_to_world(p):
    if (p[1]==400):
        p[1] = 401
    y_w =  66.73820129339674/ (p[1] - 400)
    x_w = 0.14*(p[0]-400) / (p[1] - 400)
    x_w = x_w.item()
    y_w = y_w.item()
    w = np.array([[x_w],[y_w],[0]])
    return w

def main():
    for i in range(1000000):
        x = np.random.uniform(0,10)
        y = np.random.uniform(0,10)
        w = np.array([[x],[y],[0]])
        # print(w)
        pixel = world_to_pixel(w)
        # print(pixel)
        world = pixel_to_world(pixel)
        # print(world)
        for i in range(world.shape[1]):
            if np.round(w[i],5)!= np.round(world[i],5):
                print("failure ")
                break
        # print("success")
    print("completed successfully !!")

if __name__=="__main__":
    main()