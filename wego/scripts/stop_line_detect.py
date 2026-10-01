#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import numpy as np
import cv2

class StopLineTest:
    def __init__(self):
        # rospy.init_node("stop_line_test_node")
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback)
        rospy.Subscriber("/navigation/status", Int32, self.slam_status_CB)
        self.bridge = CvBridge()
        self.stop_line_detected = False
        self.slam_status = False

        # 전체 파일 공유용 stop_line_count
        self.stop_line_count = 0    # 5일 때 로터리 도착, 6일 때 트래픽 도착, 8일 때 파이브마일 시작
        self.prev_detected = False

    def slam_status_CB(self, msg):
        self.slam_status = msg.data

    def stop_line_pub(self):
        if self.slam_status == True:
            if self.stop_line_detected and not self.prev_detected:
                self.stop_line_count += 1
                print(f"[stop_line_count]  :  {self.stop_line_count}" )
        self.prev_detected = self.stop_line_detected

    def image_callback(self, msg):
        # 1. 압축 이미지 → OpenCV 이미지
        raw_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        y, x = raw_img.shape[0:2]

        # 2. HSV 변환 → 흰색 필터링
        img_hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
        white_lower = np.array([0, 0, 192])
        white_upper = np.array([179, 64, 255])
        white_mask = cv2.inRange(img_hsv, white_lower, white_upper)
        # yellow_lower = np.array([15, 128, 0])
        # yellow_upper = np.array([40, 255, 255])
        # yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_upper)
        # ombined_range = cv2.bitwise_or(yellow_mask, white_mask)
        filtered_img = cv2.bitwise_and(raw_img, raw_img, mask=white_mask)

        # 3. 버드아이 변환
        src = np.float32([[0, 420], [275, 260], [x - 275, 260], [x, 420]])
        dst = np.float32([[x // 8, 480], [x // 8, 0], [x // 8 * 7, 0], [x // 8 * 7, 480]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(filtered_img, M, (x, y))

        # 4. Binary 변환
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        bin_img = np.zeros_like(gray)
        bin_img[gray > 50] = 1

        # 5. ROI 설정 후 흰 픽셀 수 확인
        roi = bin_img[200:280, 100:540]
        white_pixels = np.count_nonzero(roi)

        # 6. 정지선 판단
        self.stop_line_detected = white_pixels > 9000
        # if self.stop_line_detected:
        #     print(f"✅ 정지선 감지됨! 픽셀 수: {white_pixels}")
        # else:
        #     print(f"❌ 정지선 없음. 픽셀 수: {white_pixels}")

        # 7. 시각화 (디버깅용)
        # vis_img = warped.copy()
        # cv2.rectangle(vis_img, (100, 200), (540, 280), (0, 255, 0), 2)
        # cv2.imshow("Warped Image", vis_img)
        # cv2.imshow("Binary Mask", bin_img * 255)

        # self.stop_line_pub()

        # cv2.waitKey(1)

        return self.stop_line_detected


def main():
    try:
        slt = StopLineTest()
        slt.stop_line_pub()
        print(f"[stop_line_count]  :  {slt.stop_line_count}" )

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    # finally:
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
