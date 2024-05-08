#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point, Twist
from gazebo_msgs.msg import ModelStates
from math import sin, cos, pi, atan2, sqrt
from scipy.spatial.transform import Rotation
from tf.transformations import euler_from_quaternion

class GoalFollower:
    def __init__(self):
        rospy.init_node("goal_follower")
        self.curr_sub = rospy.Subscriber("/curr_goal", Point, self.goal_cb)
        self.goal_pub = rospy.Publisher("/goal", Point, queue_size = 10)
        self.model_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_cb)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 10)
        rate = rospy.Rate(10)
        self.goalX = 0
        self.goalY = 0
        self.botX = 0
        self.botY =0
        self.botOrient = 0
        self.max_dist_err = 0.17
        self.max_angle_err = 0.35
    
    def goal_cb(self,msg):
        self.goalX = msg.x
        self.goalY = msg.y

    def model_cb(self, msg):
        self.botX = msg.pose[1].position.x
        self.botY = msg.pose[1].position.y
        quat = msg.pose[1].orientation
        #temp = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        #r,p,y = temp.as_euler("XYZ", degrees= False)
        r,p,y = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.botOrient = y

    def follower(self):
        count = 0
        while not rospy.is_shutdown():
            dist_err = sqrt((self.botX - self.goalX)**2 + (self.botY - self.goalY)**2)
            if dist_err > 0.2:
                ref_angle = atan2((self.goalY - self.botY), (self.goalX - self.botX))
                if ref_angle<=0 and self.botOrient>0:
                    ref_angle+= 2*pi
                elif ref_angle>0 and self.botOrient<=0:
                    self.botOrient+=2*pi

                angle_err = ref_angle - self.botOrient
                if angle_err > pi:
                    angle_err-= 2*pi
                elif angle_err < -pi:
                    angle_err+=2*pi
                vel = Twist()
                vel.linear.x = min(dist_err, self.max_dist_err)
                angle_err = max(-self.max_angle_err, min(angle_err, self.max_angle_err))
                vel.angular.z = angle_err
                self.vel_pub.publish(vel)
            else:
                count = (count+1)%15
                goal = Point()
                goal.x = self.goalX
                goal.y = self.goalY
                self.goal_pub.publish(goal)


if __name__ == "__main__":
    try:
        goal_follower = GoalFollower()
        goal_follower.follower()

    except rospy.ROSInterruptException:
        pass
