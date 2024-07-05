import rospy
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import message_filters
import datetime

class PointCloud:
    def __init__(self):
        rospy.init_node('point_cloud')
        self.bridge = CvBridge()

        self.pc_sub = message_filters.Subscriber("/camera1/depth/points", PointCloud2)
        self.img_sub = message_filters.Subscriber("/camera1/color/image_raw", Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.pc_sub, self.img_sub], 10, 0.1)
        ts.registerCallback(self.callback)

    def callback(self, pc_msg, img_msg):
        try:
            image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        timestamp = img_msg.header.stamp.to_sec()
        formatted_timestamp = datetime.datetime.fromtimestamp(timestamp).strftime("%d%H%M%S")

        # cv2.imwrite(f"./raw_imgs3/{formatted_timestamp}.png", image)

        point_cloud = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(point_cloud))
        mask = self.detect_free_space(points)
        if image is not None:
            alpha = 0.5
            annotated = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
            cv2.imwrite(f"./seg_masks1/annotated_{timestamp}.png", annotated)
        else:
            print("error saving mask")

    def detect_free_space(self, points):
        free_space_mask = np.zeros((800, 800, 3), dtype=np.uint8)
        fx = 476.70143780997665
        fy = 476.70143780997665
        cx = 400
        cy = 400
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        T = np.array([0, 0.14, 0])
        free_space = [0, 255, 0]
        rest = [0, 0, 255]

        for point in points:
            world_coord = np.matmul(np.linalg.inv(R), (point - T))
            u, v = int((fx * point[0] / point[2]) + cx), int((fy * point[1] / point[2]) + cy)
            if not (0 <= u < 800 and 0 <= v < 800):
                continue
            if world_coord[2] < 0.01:
                free_space_mask[v, u] = free_space
            else:
                free_space_mask[v, u] = rest

        return free_space_mask

def main():
    pc = PointCloud()
    rospy.spin()

if __name__ == '__main__':
    main()
