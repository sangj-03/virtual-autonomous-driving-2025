#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import rospy, math, cv2, os, sys
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Float64, Int32
from cv_bridge import CvBridge

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)

from lane_follower import LaneFollower

class Rotary:
    def __init__(self):
        rospy.init_node("rotary_node")

        # Subscribers
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.cam_CB, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.lidar_CB, queue_size=1)
        rospy.Subscriber("/stop_line/count", Int32, self.stop_line_count_CB)

        # Publishers
        self.speed_pub = rospy.Publisher("/rotary/speed_cmd", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher("/rotary/steer_cmd", Float64, queue_size=1)
        self.speed_msg = Float64()
        self.steer_msg = Float64()

        # Buffers
        self.bridge = CvBridge()
        self.latest_img = None
        self.latest_scan = None
        self.scan_stamp = None

        # Modules
        self.lf = LaneFollower()


        # stop_line_count 공유받음
        self.stop_line_count = 0

        # 주행 파라미터(간단 상수)
        self.rate_hz = 60
        self.cruise_speed = 1200.0
        self.entry_speed = 500.0           # 진입 시 저속
        self.steer_clip_min = 0.0
        self.steer_clip_max = 1.0

        # 진입 시 목표 차선에 따른 초기 바이어스(필요시 조정)
        self.entry_bias_inner = 0.8       # 1차선(내측) 진입 시 바이어스 스티어
        self.entry_bias_outer = 0.92       # 2차선(외측) 진입 시 바이어스 스티어
        self.entry_duration = 1.5          # 진입 초반 바이어스 유지 시간[s]
        self.entry_blend_alpha = 0.7       # 바이어스:LK 혼합 가중(초기)

        # 정지선/관찰
        self.stop_hold_sec = 1.0           # 정지선에서 최소 정지 유지 시간
        self.stop_enter_ts = None
        self.target_lane = None            # 1 또는 2
        self._last_print_lane = None       # where_to_go 로그 중복 방지

        # 전방 안전 체크 각도/거리(요청: 우측 20도, 좌측 60도)
        self.front_right_deg = 15.0        # 우측 20도(음수 각 경계)
        self.front_left_deg  = 50.0        # 좌측 60도(양수 각 경계)
        self.front_safe_dist = 1.5         # 2 m 이상이면 진입 안전으로 간주 (정책 반영)F

        # where_to_go 파라미터(최소거리 구간으로 간단 분류)
        self.small_fov_deg = 5.0           # 전방 ±5도 창
        self.lane1_low, self.lane1_high = 0.85, 1.4  # 1차선 구간
        self.lane2_low, self.lane2_high = 0.1, 0.8   # 2차선 구간

        # 우측 차선 끊김 감지(탈출 트리거)
        self.exit_roi_thresh = 0.02   # ROI 내 흰 픽셀 비율 임계
        self.exit_roi_hold = 3        # 연속 프레임 유지
        self._exit_roi_cnt = 0
        self._exit_announced = False  # 우측 탈출로 최초 감지 로그 중복 방지

        # 로터리 내부 보정(진입 시 이미지 오프셋 기반)
        self.bias_k = 0.08                    # 바이어스 보정 게인
        self.speed_k = 200.0                  # 속도 보정 게인(원치 않으면 0)
        self.entry_speed_base = self.entry_speed

        # 탈출 제어
        self.exit_duration = 1.8              # 탈출 바이어스 유지 시간
        self.exit_speed = 620.0
        self.exit_bias = 1
        self.exit_start_ts = None

        # 상황 정의
        self.state = 0
        self.entry_start_ts = None

        self.rate = rospy.Rate(self.rate_hz)

    # 카메라 콜백
    def cam_CB(self, msg: CompressedImage):
        try:
            self.latest_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cam decode fail: %s", e)
            self.latest_img = None

    # 라이다 콜백
    def lidar_CB(self, msg: LaserScan):
        self.latest_scan = msg
        self.scan_stamp = msg.header.stamp if hasattr(msg, "header") and msg.header.stamp else rospy.Time.now()

    def stop_line_count_CB(self, msg):
        self.stop_line_count = msg.data

    # 정지선 검출 상태
    def stop_line_detect(self):
        if self.stop_line_count == 5:
            return True
        else: 
            return False

    # 전방 안전 체크
    # 정책: 전방 각도 창(우측 -front_right_deg ~ 좌측 +front_left_deg)에서 2.0 m 이내의 포인트만 고려.
    # - 2.0 m 이내에 측정값이 하나라도 있으면 그 최소값(min_d)을 사용하고, min_d >= 2.0 이면 안전(True).
    # - 2.0 m 이내 측정값이 전혀 없으면 '2 m 이상 확보'로 간주하여 안전(True), min_d=2.0 반환.
    def obstacle_ahead_safe(self):
        scan = self.latest_scan
        if scan is None or not hasattr(scan, "ranges") or len(scan.ranges) == 0:
            return False, float("inf")

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges, dtype=np.float32)
        valid = np.isfinite(ranges)

        right_lim = -math.radians(self.front_right_deg)  # 음수
        left_lim  =  math.radians(self.front_left_deg)   # 양수
        mask = (angles >= right_lim) & (angles <= left_lim) & valid

        if not np.any(mask):
            return False, float("inf")

        d = ranges[mask]
        if d.size == 0:
            return False, float("inf")

        # 2 m 이내만 추출
        d_within = d[d <= 1.5]

        if d_within.size == 0:
            # 2 m 이내에 아무것도 없으면 2 m 이상 확보로 간주 -> 진입 가능
            return True, 1.5

        min_d = float(np.min(d_within))
        safe = (min_d >= 1.5)
        return safe, min_d

    # 우측 차선 끊김 감지(탈출 트리거)
    # ROI에서 흰 픽셀 비율이 exit_roi_thresh보다 작아지는 상태가 exit_roi_hold 프레임 지속되면
    # 우측 차선 끊김(=우측 탈출로)으로 판단하며, 최초 판단 시 1회 로그를 출력합니다.
    def right_lane_break_detected(self):
        if self.latest_img is None:
            return False

        img = self.latest_img
        h, w = img.shape[:2]

        # ROI: 우하단 25% 폭 x 20% 높이(튜닝)
        x0 = int(w * 0.70)
        x1 = int(w * 0.95)
        y0 = int(h * 0.65)
        y1 = int(h * 0.95)
        roi = img[y0:y1, x0:x1]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 180], dtype=np.uint8)
        upper = np.array([180, 60, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        ratio = float(np.count_nonzero(mask)) / max(1, mask.size)
        broken = (ratio < self.exit_roi_thresh)

        # 히스테리시스
        prev_cnt = self._exit_roi_cnt
        if broken:
            self._exit_roi_cnt = min(self._exit_roi_cnt + 1, self.exit_roi_hold)
        else:
            self._exit_roi_cnt = max(self._exit_roi_cnt - 1, 0)

        detected = (self._exit_roi_cnt >= self.exit_roi_hold)

        # 최초 감지 시 1회성 로그 출력
        if detected and not self._exit_announced:
            rospy.loginfo("[exit] 우측 탈출로 발견")
            self._exit_announced = True

        # 감지 해제되면 플래그 리셋 (안정적인 재감지를 위해)
        if not detected and self._exit_announced and self._exit_roi_cnt == 0 and not broken:
            self._exit_announced = False

        return detected

    # 우측 차선 x-오프셋 측정(진입 보정용)
    def measure_right_lane_offset(self):
        if self.latest_img is None:
            return None

        img = self.latest_img
        h, w = img.shape[:2]
        x0 = int(w * 0.65)
        x1 = int(w * 0.98)
        y0 = int(h * 0.60)
        y1 = int(h * 0.95)
        roi = img[y0:y1, x0:x1]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 180], dtype=np.uint8)
        upper = np.array([180, 60, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None

        mean_x_roi = float(np.mean(xs))
        mean_x = x0 + mean_x_roi
        offset = np.clip(mean_x / float(w), 0.0, 1.0)  # 0(좌) ~ 1(우)
        return offset

    # 상태 전이
    def update_state(self, stop_detected):
        if self.state == 0:
            if stop_detected:
                self.state = 1
                self.stop_enter_ts = rospy.Time.now().to_sec()
                self.target_lane = None
                rospy.loginfo("[state] 정지선 감지 → 정지 및 관찰(state=1)")

        elif self.state == 1:
            # 정지 유지 시간 경과 + 전방 안전 + 목표 차선 확보 시 진입
            ready_time = False
            if self.stop_enter_ts is not None:
                now = rospy.Time.now().to_sec()
                ready_time = (now - self.stop_enter_ts) >= self.stop_hold_sec

            lane_id, conf, _ = self.where_to_go()
            if lane_id in (1, 2):
                self.target_lane = lane_id

            safe, min_d = self.obstacle_ahead_safe()

            if ready_time and safe and (self.target_lane is not None):
                self.state = 2
                self.entry_start_ts = rospy.Time.now().to_sec()
                rospy.loginfo(f"[state] 조건 만족 → 로터리 진입(state=2), lane={self.target_lane}, front_min={min_d:.2f}m")

        elif self.state == 2:
            # 우측 차선 사라짐 감지 → 탈출
            if self.right_lane_break_detected():
                # 우측 탈출로 발견 로그는 right_lane_break_detected 내부에서 이미 1회 출력됨
                self.state = 3
                self.exit_start_ts = rospy.Time.now().to_sec()
                rospy.loginfo("[state] 우측 차선 끊김 감지 → 로터리 탈출(state=3)")

        elif self.state == 3:
            # 탈출 유지 후 복귀
            if self.exit_start_ts is not None:
                now = rospy.Time.now().to_sec()
                if (now - self.exit_start_ts) >= self.exit_duration:
                    self.state = 4
                    # 리셋
                    self._exit_roi_cnt = 0
                    self._last_print_lane = None
                    self.target_lane = None
                    self._exit_announced = False
                    rospy.loginfo("[state] 탈출 완료 → 주행(state=4) 복귀")
        elif self.state == 4:
            pass

    # 차선 추종(steer 계산)
    def lane_calculate(self):
        if self.latest_img is None:
            return 0.5
        steer, _pos = self.lf.process(self.latest_img)  # (steer, pos)
        steer = float(np.clip(float(steer), self.steer_clip_min, self.steer_clip_max))
        return steer

    # 로터리 내 차량 차선 판별(전방 ±small_fov_deg에서 최소거리 구간으로 단순 판정)
    # lane_id_out: 1=1차선(내측), 2=2차선(외측), None=판단불가
    def where_to_go(self):
        scan = self.latest_scan
        if scan is None or not hasattr(scan, "ranges") or len(scan.ranges) == 0:
            return None, 0.0, {"min_d": float("inf"), "window": None}

        fov = math.radians(self.small_fov_deg)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges, dtype=np.float32)

        valid = np.isfinite(ranges)
        front_mask = (np.abs(angles) <= fov) & valid

        if not np.any(front_mask):
            rospy.loginfo(f"[where_to_go] 전방(±{self.small_fov_deg:.1f}°) 유효 포인트 없음")
            return None, 0.0, {"min_d": float("inf"), "window": None}

        d = ranges[front_mask]
        if d.size == 0:
            rospy.loginfo(f"[where_to_go] 전방(±{self.small_fov_deg:.1f}°) 유효 포인트 없음")
            return None, 0.0, {"min_d": float("inf"), "window": None}

        min_d = float(np.min(d))

        lane1_low, lane1_high = self.lane1_low, self.lane1_high
        lane2_low, lane2_high = self.lane2_low, self.lane2_high

        lane_id_out = None
        conf = 0.0
        window = None

        if lane1_low <= min_d <= lane1_high:
            lane_id_out = 1
            window = (lane1_low, lane1_high)
            center = 0.5 * (lane1_low + lane1_high)
            half_span = max(1e-6, 0.5 * (lane1_high - lane1_low))
            conf = max(0.2, 1.0 - abs(min_d - center) / half_span)
        elif lane2_low <= min_d <= lane2_high:
            lane_id_out = 2
            window = (lane2_low, lane2_high)
            center = 0.5 * (lane2_low + lane2_high)
            half_span = max(1e-6, 0.5 * (lane2_high - lane2_low))
            conf = max(0.2, 1.0 - abs(min_d - center) / half_span)
        else:
            lane_id_out = None
            conf = 0.0

        if lane_id_out is not None:
            if self._last_print_lane != lane_id_out:
                rospy.loginfo(f"[where_to_go] 전방 최소거리={min_d:.2f} m → {lane_id_out}차선 (conf={conf:.2f})")
                self._last_print_lane = lane_id_out
        else:
            #rospy.loginfo(f"[where_to_go] 전방 최소거리={min_d:.2f} m → 구간 외(판단불가)")
            pass

        metrics = {"min_d": min_d, "window": window}
        return lane_id_out, conf, metrics

    def run(self):
        while not rospy.is_shutdown():
            try:
                # self.mux_stop_line()
                # 관측 및 상태 전이
                stop_detected = self.stop_line_detect()
                self.update_state(stop_detected)

                # 상태별 제어
                if self.state == 0:
                    # 차선 중앙 주행
                    steer = self.lane_calculate()
                    speed = 0

                elif self.state == 1:
                    # 정지선에서 정지 유지 + 관찰(차선 파악/전방 안전 체크는 update_state 내부에서 수행)
                    speed, steer = 0.0, 0.5
                    self.where_to_go()
                    safe, min_d = self.obstacle_ahead_safe()
                    # rospy.loginfo_throttle(1.0, f"[front] 안전={safe}, 최소거리={min_d:.2f} m (범위: -{self.front_right_deg}° ~ +{self.front_left_deg}°)")

                elif self.state == 2:
                    # 로터리 진입: 바이어스 + LK 혼합 + 오프셋 보정
                    steer_lk = self.lane_calculate()
                    bias_nominal = self.entry_bias_inner if self.target_lane == 1 else self.entry_bias_outer

                    # 이미지 기반 우측 차선 오프셋 보정
                    offset = self.measure_right_lane_offset()
                    bias = bias_nominal
                    speed = float(self.entry_speed_base)

                    if offset is not None:
                        target_offset = 0.80  # 우측 ROI 중앙 부근
                        err = float(offset - target_offset)
                        bias = float(np.clip(bias_nominal - self.bias_k * err, self.steer_clip_min, self.steer_clip_max))
                        speed = float(np.clip(self.entry_speed_base - self.speed_k * abs(err), 500.0, self.cruise_speed))

                    if self.entry_start_ts is not None:
                        now = rospy.Time.now().to_sec()
                        if (now - self.entry_start_ts) <= self.entry_duration:
                            alpha = self.entry_blend_alpha
                            steer = float(alpha * bias + (1.0 - alpha) * steer_lk)
                        else:
                            steer = steer_lk
                    else:
                        steer = steer_lk

                elif self.state == 3:
                    # 로터리 탈출: 우회전 바이어스 유지
                    steer = float(self.exit_bias)
                    speed = float(self.exit_speed)

                elif self.state == 4:
                    steer = self.lane_calculate()
                    speed = float(self.cruise_speed)

                # Publish
                self.speed_msg.data = float(speed)
                self.steer_msg.data = float(steer)
                self.speed_pub.publish(self.speed_msg)
                self.steer_pub.publish(self.steer_msg)

            except Exception as e:
                rospy.logwarn("run loop error: %s", e)
                self.speed_msg.data = 0.0
                self.steer_msg.data = 0.5
                self.speed_pub.publish(self.speed_msg)
                self.steer_pub.publish(self.steer_msg)

            self.rate.sleep()

def main():
    try:
        node = Rotary()
        node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
