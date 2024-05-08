#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from math import sin, cos


class GoalGenerator:
    def __init__(self):
        rospy.init_node("goal_generator")
        self.reached = False
        self.goal_x = 0
        self.goal_y = 0
        self.delta = 0.19
        self.goal_sub = rospy.Subscriber("/goal", Point, self.goal_reached_cb)
        self.curr_pub = rospy.Publisher("/curr_goal", Point, queue_size = 10)
        rate = rospy.Rate(10)
        self.start_angle = 0
        self.angle = self.start_angle

    def goal_reached_cb(self,msg):
        x = msg.x
        y = msg.y
        if ((x - self.goal_x)**2 + (y - self.goal_y)**2) <= 0.3:
            self.reached = True

    def compute_goal(self):
        new_goal = Point()
        A,B,a,b = 5,3,1,2
        new_goal.x = A * cos(a*self.angle)
        new_goal.y = B * sin(b*self.angle)
        self.goal_x = new_goal.x
        self.goal_y = new_goal.y
        self.curr_pub.publish(new_goal)
        self.reached= False
        while not rospy.is_shutdown():
            if self.reached:
                self.angle += self.delta
                self.reached = False
            new_goal.x = A * cos(a*self.angle)
            new_goal.y = B * sin(b*self.angle)
            self.goal_x = new_goal.x
            self.goal_y = new_goal.y
            self.curr_pub.publish(new_goal)


if __name__ =="__main__":
    try:
        goal_generator = GoalGenerator()
        goal_generator.compute_goal()

    except rospy.ROSInterruptException:
        pass
