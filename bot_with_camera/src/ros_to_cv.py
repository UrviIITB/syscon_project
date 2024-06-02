import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torchvision.transforms as transforms
import datetime

class ConvertImage:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/turtlebot/camera1/image_raw", Image, self.callback)
        self.bridge = CvBridge()

    def callback(self,req):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req, "bgr8")
        except CvBridgeError as e:
            print(e)
        img_name = "./"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".jpg"
        cv2.imwrite(img_name, cv_image)
        print("saved image")

def main():
    rospy.init_node("ros_to_cv")
    yolop = ConvertImage()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
