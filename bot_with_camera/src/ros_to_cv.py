import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import datetime
import os
import time

img_dir = "./raw_imgs/"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

class ConvertImage:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/turtlebot/camera1/image_raw", Image, self.callback)
        self.bridge = CvBridge()

    def callback(self,req):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req, "bgr8")
        except CvBridgeError as e:
            print(e)
        img_name = datetime.datetime.now().strftime("%d%H%M%S")+".jpg"
        img_path = img_dir + img_name
        cv2.imwrite(img_path, cv_image)
        print("saved image ", img_name)
        time.sleep(1)

def main():
    rospy.init_node("ros_to_cv")
    ros_ro_cv = ConvertImage()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
