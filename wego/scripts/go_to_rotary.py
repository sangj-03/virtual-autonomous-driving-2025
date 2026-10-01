#!/usr/bin/env python3
#-*- coding:utf-8 -*-

###판단 부분부터 포매팅 신경안쓰고 작업, img warp은 추가사항 있음

import numpy as np
import cv2
import time
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge
from std_msgs.msg import Float64, Int32
import math
import tf2_ros
import tf2_geometry_msgs  
from math import *
import os

from math import *
import os


class PidCal:
    error_sum = 0
    error_old = 0
    p = [0.00275, 0.0000000001, 0.0065] # optimized kp,ki,kd
    # p = [0.001, 0.0000003, 0.003]
    dp = [p[0]/10, p[1]/12, p[2]/10] # to twiddle kp, ki, kd
    def __init__(self):
        # print "init PidCal"
        self.x = 0
        self.clear_time = time.time()

    def cal_error(self, setpoint=320):
        return setpoint - self.x

    # twiddle is for optimize the kp,ki,kd
    def twiddle(self, setpoint=320):
        best_err = self.cal_error()
        #threshold = 0.001
        #threshold = 1e-09
        threshold = 0.00000005

        # searching by move 1.1x to the target and if go more through the target comeback to -2x
        while sum(self.dp) > threshold:
            for i in range(len(self.p)):
                self.p[i] += self.dp[i]
                err = self.cal_error()

                if err < best_err:  # There was some improvement
                    best_err = err
                    self.dp[i] *= 1.2
                else:  # There was no improvement
                    self.p[i] -= 2.2*self.dp[i]  # Go into the other direction
                    err = self.cal_error()

                    if err < best_err:  # There was an improvement
                        best_err = err
                        self.dp[i] *= 1.1
                    else:  # There was no improvement
                        self.p[i] += self.dp[i]
                        # As there was no improvement, the step size in either
                        # direction, the step size might simply be too big.
                        self.dp[i] *= 0.95

    # setpoint is the center and the x_current is where the car is
    # width = 640, so 320 is the center but 318 is more accurate in real
    def pid_control(self, x_current, setpoint=320):
        self.x = x_current
        self.twiddle()

        error = setpoint - x_current
        p1 = self.p[0] * error
        self.error_sum += error
        i1 = self.p[1] * self.error_sum
        d1 = self.p[2] * (error -  self.error_old)
        self.error_old = error
        pid = p1 +i1+ d1
        return pid

class LaneFollower:
    def __init__(self) -> None:
        rospy.init_node("LKAS_Node")
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.turn_start = None
        self.turn_duration = 2.0 
        rospy.Subscriber("/navigation/status", Int32, self.slam_status_CB, queue_size=1)
        self.slam_status = False

        #Image Data
        self.cv_img = None

        self.img = None
        self.img_hsv = None
        self.y, self.x = None, None
        self.h, self.s, self.v = None, None, None
        self.img_backup = None

        #img_transform
        self.yellow_range = None
        self.white_range = None
        self.combined_range = None

        #img_warp
        self.warp_img_size = None
        self.warp_img_zoomx = None
        self.warped_img = None

        #yellow
        self.yellow_warped = None

        #sliding_window
        self.out_img = None
        self.nwindows = 12
        self.margin = 60
        self.minpix = 5
        self.threshold = 100
        self.l_lane, self.r_lane = None, None

        #판단 공통
        self.pos = None
        self.trustr = True
        self.right_lane = True
        self.stopline_toggle = 4
        self.stopline_count = 0
        self.count = 0
        self.sequence = -1
        self.cut_img_top = False
        self.yellow_based_slidingwindow = False
        self.seq_start = None
        self.elapsed_time = 0

        #lidar
        self.prev_obstacle_indices = set()
        self.obstacle_static_1 = None
        self.obstacle_static_2 = None


        #control_pub
        self.pub = rospy.Publisher("/go_to_rotary/speed_cmd", Float64, queue_size=1)
        self.pub_steer = rospy.Publisher("/go_to_rotary/steer_cmd", Float64, queue_size=1)
        self.cmd_msg = Float64()
        self.midrange = 310
        self.speed = None
        self.directControl = None
        self.pidcal = PidCal()


        #misc
        self.rate = rospy.Rate(60)
        self.bridge = CvBridge()
        self.left_traffic = False

        self.pos = None          # 제어용 위치(setpoint 역할)

    # 슬램 관련 함수
    # SLAM 상태 체크
    def slam_status_CB(self, msg):
        self.slam_status = msg.data

    # 로터리 진입 관련 함수
    def get_base_in_map(self):
            try:
                t = self.tf_buffer.lookup_transform(
                    "map", "base_link", rospy.Time(0), rospy.Duration(0.2)
                )
                return t.transform.translation.x, t.transform.translation.y
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"TF lookup failed: {e}")
                return None

    def is_reached_before_ratary(self, thr=0.5):
            wp_x, wp_y = 31.004663415273333, 0.703791198730469
            pose = self.get_base_in_map()
            # print(pose)
            if pose is None:
                return False, None
            dist = math.hypot(pose[0] - wp_x, pose[1] - wp_y)
            # print(dist)
            return dist < thr, dist

    ##인지
    def img_init(self, img):# -> img,img_hsv,x,y,h,s,v
        self.img = img
        self.y, self.x, _ = img.shape
        self.img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.h, self.s, self.v = cv2.split(self.img_hsv)

        #warp_img
        self.warp_img_size = [self.x,self.x]
        self.warp_img_zoomx = self.x//4

    def img_transform(self): # img_hsv -> img
        img_hsv = self.img_hsv

        ## 노란선 흰선 분리
        lower_yellow = np.array([15, 100, 140])
        upper_yellow = np.array([30, 200, 255])
        self.yellow_range = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        lower_white = np.array([0, 0, 140])
        upper_white = np.array([50, 70, 255])
        self.white_range = cv2.inRange(img_hsv, lower_white, upper_white)

        ## 합치기
        self.combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)
        # filtered_image = cv2.bitwise_and(self.cv_img, self.cv_img, mask = combined_range)

        self.img = self.combined_range

    def img_warp(self, img=None, change_img=True, warp_img_zoomx=None): # warp_img_size,warp_img_zoomx,img,x,y -> img,x,y
        y, x = self.y, self.x
        if img is None: img = self.img
        warp_img_size = self.warp_img_size
        if warp_img_zoomx is None:warp_img_zoomx = self.warp_img_zoomx

        # Warp img to ROI(warped img)
        # topx = 269
        # bottomx = -25
        # topy = 271
        # bottomy = y
        # src_points = np.float32([[bottomx, bottomy], [topx, topy], [x - topx, topy], [x-bottomx, bottomy]])

        src_point1 = [0,450]    # 왼쪽 아래
        src_point2 = [285,260]  # 왼쪽 위
        src_point3 = [x-285,260]    # 오른쪽 위
        src_point4 = [x,450]        # 오른쪽 아래
        src_points = np.float32([src_point1,src_point2,src_point3,src_point4])

        # print(src_points)
        topx = warp_img_zoomx
        bottomx = warp_img_zoomx
        topy = x//4
        bottomy = x
        dst_points = np.float32([[bottomx, bottomy], [topx, topy], [x - topx, topy], [x-bottomx, bottomy]])
        # print(dst_points)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        # print(matrix)

        self.warped_img = cv2.warpPerspective(img, matrix, warp_img_size)

        if change_img:
            self.img = self.warped_img
            self.x, self.y = warp_img_size

    def sliding_window(self): # nwindows,margin,minpix,threshold,midpoint,img -> l_lane,r_lane
        nwindows = self.nwindows    # 윈도우 개수
        margin = self.margin        # 윈도우 좌우 폭
        minpix = self.minpix        # 픽셀 수 임계값
        lane = self.img             # 이진화된 차선 이미지
        midpoint = self.x//2        # 이미지 가로 중앙
        threshold = self.threshold

        histogram = np.sum(lane, axis=0)

        leftx_current = np.argmax(histogram[:midpoint])
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(self.x/nwindows)
        nz = lane.nonzero() # nz[0] is vertical(y), nz[1] is horizontal(x)

        left_lane_inds = []
        right_lane_inds = []

        lx, ly, rx, ry = [], [], [], []

        self.out_img = np.dstack((lane, lane, lane))*255

        l_err = [0,0]
        r_err = [0,0]
        foundr, foundl = (False, False)

        for window in range(nwindows):

            win_yl = self.x - (window+1)*window_height
            win_yh = self.x - window*window_height

            # 오른쪽 / 왼쪽 윈도우 범위 설정
            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            cv2.rectangle(self.out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) 
            cv2.rectangle(self.out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2) 

            # 윈도우 내의 차선 픽셀 인덱스 추출
            good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
            good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # 최소 픽셀 이상(하얀색이 충분하면) 그 x좌표 평균을 새 중심으로 업데이트
            if len(good_left_inds) > minpix:
                if leftx_current > threshold and leftx_current < histogram.shape[0]-threshold:
                    l_err[1] = 0
                    foundl = True
                else: l_err[1] += 1
                leftx_current = np.int(np.mean(nz[1][good_left_inds]))
            else: l_err[1] += 1
            if not foundl: l_err[0] += 1
            if len(good_right_inds) > minpix:
                if rightx_current > threshold and rightx_current < histogram.shape[0]-threshold:
                    r_err[1] = 0
                    foundr = True
                else: r_err[1] += 1
                rightx_current = np.int(np.mean(nz[1][good_right_inds]))
            else: r_err[1] += 1
            if not foundr: r_err[0] += 1

            # 차선위치 x좌표 좌/우 & 슬라이딩 윈도우 수직 위치
            lx.append(leftx_current)
            ly.append((win_yl + win_yh)/2)

            rx.append(rightx_current)
            ry.append((win_yl + win_yh)/2)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)


        # 왼쪽은 파랑, 오른쪽은 빨강
        self.out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        self.out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]

        # 결과 저장
        self.l_lane, self.r_lane = (lx, ly, l_err), (rx, ry, r_err)

    ##판단

    def adjust_img(self):
        if self.cut_img_top:
            if np.sum(self.img[:][:self.y//2]) > 7000000:
                self.cut_img_top = None
                print("///////////")
            self.img[:][:self.y//2] = 0
        if self.yellow_based_slidingwindow:
            self.img_backup = self.img
            self.y, self.x = self.yellow_range.shape
            self.img_warp(img=self.yellow_range)

    def go_forward(self):
        poslf, posrf = self.l_lane, self.r_lane
        x = self.x
        line_len = 286

        posl = int(poslf[0][3])
        posr = int(posrf[0][3])
        fail_threshold = 3
        # print(posr[2])
        if max(poslf[2]) >= fail_threshold: posl = posr - line_len 

        if not self.trustr:
            if max(posrf[2]) >= fail_threshold: posr = posl + line_len

            if posl == posr:
                posl = x//2

        self.pos = (posl + posr) // 2

    def go_left(self, max_offset = 200):

         now = rospy.get_time()
        # 회전 시작 타임 설정
         if self.turn_start is None:
            self.turn_start = now

        # 경과 시간 비율 (0.0 ~ 1.0)
         elapsed = now - self.turn_start
         ratio = min(elapsed / self.turn_duration, 1.0)

        # 현재 오프셋: 0 → max_offset
         current_offset = max_offset * ratio

        # 슬라이딩 윈도우로 차선 정보 갱신
         self.sliding_window()

        # 기본 pos 계산 (이전과 동일)
         posl = int(self.r_lane[0][3])
         line_len = 270
         posl += current_offset * 0.8
         self.pos = (posl*2 + line_len)/2

        # 회전 완료 시 turn_start 초기화
         if ratio >= 1.0:
            self.turn_start = None

    def go_sequence(self):

        if self.sequence == -1:
            self.speed = 1800
            self.go_forward()
            self.start_time=time.time()
            reached, dist = self.is_reached_before_ratary()
            if reached:
                print("Waypoint reached")
                self.sequence = 8
            else:
                # print(f"Driving... dist left {dist:.2f}m")
                pass
            self.obstacle_static = None

        elif self.sequence == -2:
            self.speed = 1350
            self.go_forward()

        # 동적 장애물: 정지 1.5초
        elif self.sequence == 0:
            now = rospy.get_time()
            self.stop_start_time = getattr(self, 'stop_start_time', None)
            if self.stop_start_time is None:
                self.stop_start_time = now
            self.speed = 0
            if now - self.stop_start_time >= 1.5:
                self.obstacle_static_1 = None
                self.obstacle_static_2 = None
                self.sequence = -1
                self.stop_start_time = None

        # 정적 장애물 인식 = 지그재그                
        elif self.sequence == 1 and self.right_lane == True:
            now_1 = rospy.get_time()
            if self.seq_start is None:
                self.seq_start = now_1

            self.elapsed_time = now_1 - self.seq_start

            if self.right_lane == True:
                if self.elapsed_time > 0.8:
                    self.right_lane = False
                    self.obstacle_static_1 = None
                    self.obstacle_static_2 = None
                    self.elapsed_time = 0
                    self.sequence = 2
                    self.seq_start = None

        elif self.right_lane == False and self.sequence == 1 :

            now_2 = rospy.get_time()
            if self.seq_start is None:
                self.seq_start = now_2

            self.elapsed_time = now_2 - self.seq_start    

            if self.elapsed_time > 1.0:
                self.right_lane = True
                self.obstacle_static_1 = None
                self.obstacle_static_2 = None
                self.elapsed_time = 0
                self.sequence = 2
                self.seq_start = None
        # 보정띠
        elif self.sequence == 2 :

            now = rospy.get_time()
            if self.seq_start is None:
                self.seq_start = now
            elapsed_time = now - self.seq_start
            if elapsed_time > 1.3 :
                self.seq_start = None
                self.sequence = 3

        elif self.sequence == 3:
            self.speed = 500
            self.go_forward()
            now = rospy.get_time()
            if self.seq_start is None:
                self.seq_start = now
            elapsed_time = now - self.seq_start
            if elapsed_time > 0.6:
                self.seq_start = None
                self.sequence = -1

        elif self.sequence == 8:
            print("[LKAS_Node] : 로터리 진입 시작")
            elapsed_time = time.time() - float(self.start_time)
            if elapsed_time < 1.15:
                self.directControl = 0.22
            else:
                self.directControl = None
                self.go_left()
                self.sequence = -2

    def obstacle_dicide(self, msg):
        self.scan_msg = msg
        degree_min = self.scan_msg.angle_min * 180/pi
        degree_angle_increment = self.scan_msg.angle_increment * 180/pi
        degrees = [degree_min + degree_angle_increment * idx for idx, _ in enumerate(msg.ranges)]

        if self.sequence == 1 or self.sequence == 2 or self.sequence == -2 or self.sequence == 8 or self.sequence == 3: 
            return

        else :
            # 정적 장애물 1차 걸러내기
            current_obstacles_1 = set()
            for i, r in enumerate(msg.ranges):
                if abs(degrees[i]) < 1 and 0 < r < 1.5:
                    current_obstacles_1.add(i)
                    self.obstacle_static_1 = True

            # 동적장애물 걸러내기 & 정적 장애물 2차 걸러내기
            current_obstacles_2 = set()
            for i, r in enumerate(msg.ranges):
                if abs(degrees[i]) < 12 and 0 < r < 2:
                    current_obstacles_2.add(i)

            # 3초 전 데이터 없으면 초기화
            now = time.time()
            if not hasattr(self, 'obstacle_history'):
                self.obstacle_history = {'time': now, 'indices': current_obstacles_2}
                return

            if now - self.obstacle_history['time'] >= 0.2:
                if current_obstacles_2 and self.obstacle_history['indices']:
                    prev_center = sum(self.obstacle_history['indices']) / len(self.obstacle_history['indices'])
                    curr_center = sum(current_obstacles_2) / len(current_obstacles_2)
                    diff = abs(curr_center - prev_center)
                    print(diff)
                    if self.obstacle_static_1 == True and self.obstacle_static_2 == True:

                        pass

                    elif diff > 0.55:  
                        print("🚨 동적 장애물 가능성")
                        self.obstacle_static_2 = False

                    else :
                        self.obstacle_static_2 = True 
                        print("🧱 정적 장애물 가능성")

                # 기록 갱신
                self.obstacle_history = {'time': now, 'indices': current_obstacles_2}

            if self.sequence == 1 or self.sequence == 2 :
                self.obstacle_static_2 != False

            # 그냥 동적장애물 일 때
            if self.obstacle_static_2 == False :
                self.sequence = 0

            # 완전한 정적장애물 일 때
            elif self.obstacle_static_2 == True and self.obstacle_static_1 == True :
                self.sequence = 1

            # 걍 둘 중 하나일 때
            else :
                pass

    ##제어
    def control_pub(self, ctrl = None): # speed,midrange,x,pos
        midrange = self.midrange
        x = self.x
        pos = self.pos

        if self.sequence == 1:
                if self.right_lane == True:
                    # print("왼쪽 가즈아")
                    ctrl = 0.0
                    self.speed = 570

                elif self.right_lane == False:
                    # print("오른쪽 가즈아")
                    ctrl = 1
                    self.speed = 570

        elif self.sequence == 2:
                if self.right_lane == False:
                    # print("왼보정")
                    pid = 1.5
                    self.speed = 500
                elif self.right_lane == True:
                    # print("우보정")
                    self.speed = 500
                    pid = 0.5


        else :
            pid = self.pidcal.pid_control(self.pos,setpoint=self.x//2)

        if ctrl == None :
            ctrl = abs(pid - 0.5)

        self.cmd_msg.data = ctrl #조향각 설정
        self.pub_steer.publish(self.cmd_msg.data)
        self.cmd_msg.data = self.speed
        self.pub.publish(self.cmd_msg.data)

    ## ROS 관련 메서드
    def subscribe(self):
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self._img_cb, queue_size=1)
        rospy.Subscriber("/scan",LaserScan,self.lidar_CB)
    def _img_cb(self, msg):
        if self.slam_status == True:
            self.cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
    def lidar_CB(self,msg):
        if self.slam_status == True:
            self.obstacle_dicide(msg)

if __name__ == "__main__":
    car = LaneFollower()
    car.subscribe()
    while not rospy.is_shutdown():
        while car.cv_img is None: car.rate.sleep()

        if car.slam_status == True:
            car.img_init(car.cv_img)
            car.img_transform()
            car.img_warp(car.yellow_range, change_img=False, warp_img_zoomx=car.x//2.5)
            car.yellow_warped = car.warped_img
            car.img_warp()
            car.adjust_img()#판단
            car.sliding_window()

            #판단
            car.go_sequence()
            #제어
            car.control_pub(ctrl=car.directControl)

            car.rate.sleep()
