#!/usr/bin/env python3
import rospy
import numpy as np
from math import cos, sin, pi
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tf.transformations import euler_from_quaternion

## plot settings
Ylimit = 6
Xlimit = 6


class DynamicUpdate():
    def __init__(self, fig, ax):
        self.xdata=[]
        self.ydata=[]
        self.theta=[]

        self.fig, self.ax = fig, ax
        
        self.li,  = plt.plot(self.xdata, self.ydata, 'b:', lw=2.0)
        self.head,  = plt.plot(self.xdata, self.ydata, 'k', lw=1.2)
        self.bot_c = Circle((0, 0), 0.3, fc='g', alpha=0.5)
        self.way_pt = Circle((0, 0), 0.2, fc='r')
        self.ax.add_patch(self.bot_c)
        self.ax.add_patch(self.way_pt)

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
        self.ax.grid()

    def waypoint_callback(self, data):
        '''update the waypoint'''
        self.way_pt.center = data.x, data.y

    def odom_callback(self, data):
        
        temp = data.pose[1] ## This is hard-coded param

        x_r = temp.position.x
        y_r = temp.position.y
        

        x= temp.orientation.x
        y= temp.orientation.y
        z= temp.orientation.z
        w= temp.orientation.w

        roll, pitch, yaw = euler_from_quaternion([x,y,z,w])
        
        if (len(self.xdata) and len(self.ydata)):
            ## check the distance
            if ((self.xdata[-1]-x_r)**2 + (self.ydata[-1]-y_r)**2)**(0.5) > 0.1 :
                self.xdata.append(x_r)
                self.ydata.append(y_r)
            
                self.theta.append(yaw)
        else:
            self.xdata.append(x_r)
            self.ydata.append(y_r)
            self.theta.append(yaw)

        self.li.set_xdata(self.xdata)
        self.li.set_ydata(self.ydata)

        self.bot_c.center = x_r, y_r
        self.head.set_xdata([x_r, x_r+0.2*cos(yaw)])
        self.head.set_ydata([y_r, y_r+0.2*sin(yaw)])

    def update(self):
        #self.ax.relim()
        #self.ax.autoscale_view()
        #We need to draw *and* flush
        print(" ")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



def process(bot_name):


    fig, ax = plt.subplots(1,1,figsize=(5,5))
    plot = DynamicUpdate(fig, ax)
    
#!/usr/bin/env python
import rospy
import numpy as np
from math import cos, sin, pi
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tf.transformations import euler_from_quaternion

## plot settings
Ylimit = 6
Xlimit = 6


class DynamicUpdate():
    def __init__(self, fig, ax):
        self.xdata=[]
        self.ydata=[]
        self.theta=[]

        self.fig, self.ax = fig, ax
        
        self.li,  = plt.plot(self.xdata, self.ydata, 'b:', lw=2.0)
        self.head,  = plt.plot(self.xdata, self.ydata, 'k', lw=1.2)
        self.bot_c = Circle((0, 0), 0.3, fc='g', alpha=0.5)
        self.way_pt = Circle((0, 0), 0.2, fc='r')
        self.ax.add_patch(self.bot_c)
        self.ax.add_patch(self.way_pt)

        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
        self.ax.grid()

    def waypoint_callback(self, data):
        '''update the waypoint'''
        self.way_pt.center = data.x, data.y

    def odom_callback(self, data):
        
        temp = data.pose[1] ## This is hard-coded param

        x_r = temp.position.x
        y_r = temp.position.y
        

        x= temp.orientation.x
        y= temp.orientation.y
        z= temp.orientation.z
        w= temp.orientation.w

        roll, pitch, yaw = euler_from_quaternion([x,y,z,w])
        
        if (len(self.xdata) and len(self.ydata)):
            ## check the distance
            if ((self.xdata[-1]-x_r)**2 + (self.ydata[-1]-y_r)**2)**(0.5) > 0.1 :
                self.xdata.append(x_r)
                self.ydata.append(y_r)
            
                self.theta.append(yaw)
        else:
            self.xdata.append(x_r)
            self.ydata.append(y_r)
            self.theta.append(yaw)

        self.li.set_xdata(self.xdata)
        self.li.set_ydata(self.ydata)

        self.bot_c.center = x_r, y_r
        self.head.set_xdata([x_r, x_r+0.2*cos(yaw)])
        self.head.set_ydata([y_r, y_r+0.2*sin(yaw)])

    def update(self):
        #self.ax.relim()
        #self.ax.autoscale_view()
        #We need to draw *and* flush
        print(" ")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



def process(bot_name):


    fig, ax = plt.subplots(1,1,figsize=(5,5))
    plot = DynamicUpdate(fig, ax)
    

    rospy.init_node('{}_con'.format(bot_name), anonymous=True)
    rospy.Subscriber('/gazebo/model_states', ModelStates, plot.odom_callback)
    rospy.Subscriber('/curr_goal', Point, plot.waypoint_callback)
    #rospy.Subscriber('/global_pose', String, PO.callback)
    #pub = rospy.Publisher('/{}/cmd_vel'.format(bot_name), Twist, queue_size=10)
    rate = rospy.Rate(3) # 10hz
    lopcount = 0
    plt.xlim([-Xlimit, Xlimit])
    plt.ylim([-Ylimit, Ylimit])
    while not rospy.is_shutdown():
        plot.update()
        rate.sleep()


if __name__ == '__main__':
    print("Process started ")
    plt.ion()
    process('bot_1')
    rospy.init_node('{}_con'.format(bot_name), anonymous=True)
    rospy.Subscriber('/gazebo/model_states', ModelStates, plot.odom_callback)
    rospy.Subscriber('/current_goal', Point, plot.waypoint_callback)
    #rospy.Subscriber('/global_pose', String, PO.callback)
    #pub = rospy.Publisher('/{}/cmd_vel'.format(bot_name), Twist, queue_size=10)
    rate = rospy.Rate(3) # 10hz
    lopcount = 0
    plt.xlim([-Xlimit, Xlimit])
    plt.ylim([-Ylimit, Ylimit])
    while not rospy.is_shutdown():
        plot.update()
        rate.sleep()


if __name__ == '__main__':
    print("Process started ")
    plt.ion()
    process('bot_1')
