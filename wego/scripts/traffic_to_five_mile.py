#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2
import time, os, sys
import rospy
from sensor_msgs.msg import CompressedImage, Imu
from morai_msgs.msg import GetTrafficLightStatus
from cv_bridge import CvBridge
from std_msgs.msg import Float64, Int32
from math import pi, atan2

# ===================== PID (Lane follow용) =====================
class PidCal:
    error_sum = 0.0
    error_old = 0.0
    # kp, ki, kd
    p = [0.0021, 2.0e-6, 0.000]

    def __init__(self):
        self.x = 0.0
        self._d_freeze_until = 0.0  # 시퀀스 전환 직후 D 감쇠용

    def reset(self):
        self.error_sum = 0.0
        self.error_old = 0.0

    def pid_control(self, x_current, setpoint=320.0):
        if x_current is None:
            return 0.0
        self.x = float(x_current)
        error = float(setpoint) - float(x_current)
        p1 = self.p[0] * error
        self.error_sum += error
        i1 = self.p[1] * self.error_sum
        derr = error - self.error_old
        if rospy.get_time() < self._d_freeze_until:
            derr = 0.0
        d1 = self.p[2] * derr
        self.error_old = error
        return p1 + i1 + d1
# ===============================================================

class IntegratedLeftTurnAndLane:
    def __init__(self):
        rospy.init_node("traffic_to_five_mile_node")

        # --- Publishers
        self.speed_pub = rospy.Publisher("/ttfm/speed_cmd", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher("/ttfm/steer_cmd", Float64, queue_size=1, latch=True)
        self.speed_msg = Float64()
        self.steer_msg = Float64()

        # --- Subscribers
        img_topic = rospy.get_param('~img_topic', '/image_jpeg/compressed')
        imu_topic = rospy.get_param('~imu_topic', '/imu')
        # rospy.loginfo(f"[SUB] img={img_topic} imu={imu_topic}")
        rospy.Subscriber(img_topic, CompressedImage, self._img_cb, queue_size=1)
        rospy.Subscriber(imu_topic, Imu, self.imu_cb, queue_size=1)
        rospy.Subscriber("/GetTrafficLightStatus", GetTrafficLightStatus, self._tl_cb, queue_size=1)
        rospy.Subscriber("/navigation/status", Int32, self.slam_status_CB, queue_size=1)
        rospy.Subscriber("/stop_line/count", Int32, self.stop_line_CB, queue_size=1)

        self.slam_status = False
        self.stop_line_count = 0

        # --- Traffic/Turn FSM
        self.signal = 0
        # WAIT → WAIT_HOLD → TURN → HANDOFF → FOLLOW
        self.mode = "WAIT"
        self.wait_hold_start = None
        self.turn_start = None

        # WAIT_HOLD: 신호 수신 후 고정 대기(항상 동일 조건에서 TURN 시작)
        self.WAIT_HOLD_TIME = 0.60  # 0.5~1.0s 권장

        # TURN 안정화 패치
        self.SLEW_BYPASS_IN_TURN = True  # TURN 중 슬루 우회(조향 즉시 적용)
        self.TURN_PRE_HOLD = 0.20        # TURN 시작 직후 정지, 조향 먼저 세팅

        # TURN 파라미터(오픈루프 좌회전)
        self.TURN_SPEED = 1500
        self.TURN_TIME  = 2.65
        self.STEER_CENTER = 0.5
        self.STEER_LEFT = 0.322       # 극성 반대면 INVERT_STEER=True
        self.INVERT_STEER = False

        # 조향 슬루(평상시)
        self.SLEW_UP = 0.04
        self.SLEW_DOWN = 0.03
        self.prev_steer = 0.5

        # --- Perception common
        self.bridge = CvBridge()
        self.cv_img = None
        self.img = None
        self.img_hsv = None
        self.y, self.x = None, None  # H, W
        self.h, self.s, self.v = None, None, None

        # masks / warp
        self.yellow_range = None
        self.white_range = None
        self.combined_range = None
        self.yellow_warped = None
        self.white_warped = None
        self.warp_img_size = None
        self.warp_img_zoomx = None
        self.warped_img = None

        # --- Lane follow states
        self.pos = None
        self.last_pos = None
        self.sequence = -1  # -1: 기본 노란선 추종
        self.stopline_count = 0
        self.speed = 0.0
        self.directControl = None
        self.pidcal = PidCal()

        # ▶ 노란선에서 중앙까지 오프셋 (1차선 유지)
        self.lane_half_px = rospy.get_param('~lane_half_px', 40)

        # IMU
        self.yaw = None
        self.yaw_alpha = 0.10
        self.k_yaw_st = 0.30
        self.straight_yaw_target = 1.598   # ≈91.6°
        self.imu_deadband = np.deg2rad(0.0)
        self.imu_ready = False
        self._yaw_ref_pi_for_100 = False
        self.yaw_meas_offset = 0.0
        self.yaw_ref_add_pi = False
        self.imu_err_sign = 1.0

        # 종료 처리(스티어 0° 유지)
        self.shutdown_steer_value = 0.5
        self.shutdown_publish_repeats = 20
        self.shutdown_publish_dt = 0.05
        self.exiting = False

        # 우회전 오픈루프(원본 유지)
        self.right_turn_speed = 1800
        self.right_turn_profile = [(0.0, 0.95), (0.90, 0.90), (1.6, 0.90)]
        self.right_turn_total = 1.15

        # 두 번째 정지선 직진(IMU 보조)
        self.second_straight_duration = 1.5
        self.second_straight_speed = 1800
        self.straight2_start = None
        self.straight_imu_on = False
        self.straight_imu_window = 1.2
        self.straight_imu_start = None

        # 좌회전 후 스톱라인 무시 시간 & 핸드오프 시각
        self._ignore_stopline_until = 0.0
        self._handoff_ts = 0.0

        # ==== 패치 B: stop_line 빠른 디바운스 파라미터 ====
        self.STOP_BAND = (0.84, 0.98)   # 하단 84%~98%만 검사
        self.STOP_MIN_FRAC = 0.12       # 한 줄 최소 흰 픽셀 비율(기존 0.20→0.12)
        self.STOP_REQ_FRAMES = 3        # 연속 합격 프레임 수(2~3 권장)
        self.STOP_COOLDOWN = 0.5        # 재트리거 쿨다운
        self._stop_det_frames = 0
        self._last_stop_ts = 0.0

        # ==== 정지선 액션 매핑(4개) ====
        # 1: 우회전, 2: 직진(IMU), 3: 패스, 4: 종료
        self.STOP_ACTIONS = {
            1: "right_turn",
            2: "straight_imu",
            4: "exit",
        }

        self.boot_start = time.monotonic()
        self.boot_armed = False
        self.img_ready = False
        self.tl_ready = False
        self.MIN_BOOT_HOLD = 0.30
        self._allow_left_prev = False

        rospy.on_shutdown(self._on_shutdown)
        self.rate = rospy.Rate(60)

    def slam_status_CB(self, msg):
        self.slam_status = msg.data

    def stop_line_CB(self, msg):
        self.stop_line_count = msg.data

    # ================= ROS Callbacks =================
    def _img_cb(self, msg):
        if self.stop_line_count >= 6:
            self.cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
            self.img_ready = True

    def _tl_cb(self, msg: GetTrafficLightStatus):
        if self.stop_line_count >= 6:
            self.signal = msg.trafficLightStatus
            self.tl_ready = True

    def imu_cb(self, raw: Imu):
        if self.stop_line_count >= 6:
            try:
                qx = float(raw.orientation.x)
                qy = float(raw.orientation.y)
                qz = float(raw.orientation.z)
                qw = float(raw.orientation.w)
            except Exception:   
                return
            arr = np.array([qx, qy, qz, qw], dtype=float)
            if not np.isfinite(arr).all(): return
            n2 = float(np.dot(arr, arr))
            if n2 < 1e-9: return
            if not (0.5 < n2 < 1.5):
                arr *= (1.0 / np.sqrt(n2))
            qx, qy, qz, qw = arr.tolist()
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = atan2(siny_cosp, cosy_cosp)
            if self.yaw is None:
                self.yaw = yaw
            else:
                self.yaw = (1.0 - self.yaw_alpha) * self.yaw + self.yaw_alpha * yaw
            self.imu_ready = True

    # ================= Perception (Lane pipeline) =================
    def img_init(self, img):
        if self.stop_line_count >= 6:
            self.img = img
            self.y, self.x = img.shape[:2]          # H, W
            self.img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self.h, self.s, self.v = cv2.split(self.img_hsv)
            self.warp_img_size = (self.x, self.x)   # (W, W)
            self.warp_img_zoomx = self.x // 4

    def img_transform(self):
        if self.stop_line_count >= 6:
            img_hsv = self.img_hsv
            # 노란선
            lower_yellow = np.array([15, 100, 140], dtype=np.uint8)
            upper_yellow = np.array([30, 200, 255], dtype=np.uint8)
            self.yellow_range = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

            # 흰색 (정지선) + 패치 C: 닫기(CLOSE) 후 팽창(DILATE)로 두껍게
            lower_white  = np.array([0, 0, 140], dtype=np.uint8)
            upper_white  = np.array([50, 70, 255], dtype=np.uint8)
            white = cv2.inRange(img_hsv, lower_white, upper_white)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k, iterations=1)
            white = cv2.dilate(white, k, iterations=1)

            self.white_range = white
            self.combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)
            self.img = self.combined_range

    def img_warp(self, img=None, change_img=True, warp_img_zoomx=None):
        if self.stop_line_count >= 6:
            y, x = self.y, self.x
            if img is None: img = self.img
            warp_img_size = self.warp_img_size
            if warp_img_zoomx is None: warp_img_zoomx = self.warp_img_zoomx

            # 원근변환 포인트(원본 유지)
            src_points = np.float32([[0, 450], [285, 260], [x - 285, 260], [x, 450]])
            topx = warp_img_zoomx
            bottomx = warp_img_zoomx
            topy = x // 4
            bottomy = x
            dst_points = np.float32([[bottomx, bottomy], [topx, topy], [x - topx, topy], [x - bottomx, bottomy]])
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            self.warped_img = cv2.warpPerspective(img, matrix, warp_img_size)

            if change_img:
                self.img = self.warped_img
                self.x, self.y = warp_img_size  # (W, H) = (W, W)

    # 노란선 히스토그램 기반 center pos 계산 (곡률 적응형 + 중앙 보정)
    def go_yellow(self):
        if self.stop_line_count >= 6:
            if self.yellow_warped is None:
                return
            H, W = self.yellow_warped.shape[:2]

            col_sum = np.sum(self.yellow_warped, axis=0).astype(np.float64)
            s = col_sum.sum()
            if s < 1e-6:
                return
            x_all = (np.arange(W) * col_sum).sum() / s

            def band_centroid(img, y0, y1, fallback):
                band = img[y0:y1, :]
                col = np.sum(band, axis=0).astype(np.float64)
                ss = col.sum()
                if ss < 1e-6:
                    return fallback
                return (np.arange(col.shape[0]) * col).sum() / ss

            y_low0,  y_low1  = int(0.78 * H), int(0.96 * H)
            y_high0, y_high1 = int(0.40 * H), int(0.58 * H)
            x_low  = band_centroid(self.yellow_warped, y_low0,  y_low1,  x_all)
            x_high = band_centroid(self.yellow_warped, y_high0, y_high1, x_all)

            curv = abs(x_high - x_low)
            if not hasattr(self, "_curv_ema"):
                self._curv_ema = curv
            else:
                self._curv_ema = 0.6 * self._curv_ema + 0.4 * curv

            lane_half_px_now = float(rospy.get_param("~lane_half_px", self.lane_half_px))  # 기본 40
            center_boost_px  = float(rospy.get_param("~center_boost_px", 12.0))            # 항상 + 보정
            base_offset = lane_half_px_now + center_boost_px

            gain  = float(rospy.get_param("~curve_offset_gain", 0.40))
            extra = float(np.clip(gain * self._curv_ema, 0.0, 35.0))

            alpha_scale = float(rospy.get_param("~preview_alpha_scale", 50.0))
            alpha = float(np.clip(self._curv_ema / alpha_scale, 0.0, 0.9))
            x_preview = (1.0 - alpha) * x_low + alpha * x_high

            self.pos = float(np.clip(x_preview + base_offset + extra, 0, W - 1))
            self.last_pos = self.pos

    # ================= Control/Sequence (Lane pipeline) =================
    def control_pub(self, ctrl=None):
        if self.stop_line_count >= 6:
            if self.exiting:
                self._pub_all(self.shutdown_steer_value, self.speed)
                return

            if ctrl is None:
                sp = (self.x // 2) if isinstance(self.x, int) else 320
                meas = self.pos if self.pos is not None else (self.last_pos if self.last_pos is not None else sp)
                pid = self.pidcal.pid_control(meas, setpoint=sp)
                ctrl = 0.5 - pid  # 부호 주의
                ctrl = float(np.clip(ctrl, 0.0, 1.0))

            self._pub_all(ctrl, float(self.speed))

    def _pub_all(self, steer_norm, motor_speed):
        if self.stop_line_count >= 6:
            self.steer_msg.data = float(steer_norm)
            self.steer_pub.publish(self.steer_msg)
            self.speed_msg.data = float(motor_speed)
            self.speed_pub.publish(self.speed_msg)

    def go_sequence(self):
        if self.stop_line_count >= 6:
            if self.sequence == -1:
                self.speed = 1800
                self.go_yellow()

            elif self.sequence == 0:
                self.speed = 0

            elif self.sequence == 91:
                # 우회전 대기(중립 후 진입)
                self.speed = self.right_turn_speed
                self.directControl = 0.5
                if time.time() - self.turn_delay_start >= 0.10:
                    self.directControl = None
                    self.sequence = 90
                    self.start_time = time.time()
                    self.speed = self.right_turn_speed

            elif self.sequence == 90:
                # 우회전 오픈루프
                t = time.time() - float(self.start_time)
                steer = self.right_turn_profile[-1][1]
                for th, val in self.right_turn_profile:
                    if t < th:
                        steer = val
                        break
                self.speed = self.right_turn_speed
                self.directControl = steer
                if t >= self.right_turn_total:
                    # 턴 종료 → PID 복귀
                    self.directControl = None
                    self.sequence = -1
                    self.pidcal.reset()
                    self.pidcal._d_freeze_until = rospy.get_time() + 0.1
                    self._ignore_stopline_until = rospy.get_time() + 1.2

            elif self.sequence == 100:
                # 직진 루프(IMU 보조)
                now = time.time()
                if self.straight2_start is None:
                    self.straight2_start = now
                    self.straight_imu_on = False
                    self.straight_imu_start = None
                    self._yaw_ref_pi_for_100 = False
                    self.yaw_ref = float(self.straight_yaw_target)
                    self.yaw_meas_offset = 0.0
                    rospy.loginfo(f"[STRAIGHT] armed. yaw_ref={self.yaw_ref:.4f} rad. waiting IMU...")

                if (not self.straight_imu_on) and self.imu_ready and (self.yaw is not None):
                    yaw_m_now = self._wrap_pi(self.yaw + self.yaw_meas_offset)
                    err0 = abs(self._wrap_pi(yaw_m_now - self.yaw_ref))
                    err1 = abs(self._wrap_pi(yaw_m_now - (self.yaw_ref + np.pi)))
                    self._yaw_ref_pi_for_100 = (err1 < err0)
                    if self._yaw_ref_pi_for_100:
                        rospy.loginfo("[STRAIGHT IMU] auto π alignment: using ref+π")
                    self.straight_imu_on = True
                    self.straight_imu_start = now
                    rospy.loginfo("[STRAIGHT IMU] window START")

                neutral = 0.5
                use_imu_now = (
                    self.straight_imu_on and
                    (self.straight_imu_start is not None) and
                    ((now - self.straight_imu_start) < self.straight_imu_window) and
                    (self.yaw is not None)
                )
                if use_imu_now:
                    yaw_m = self._wrap_pi(self.yaw + self.yaw_meas_offset)
                    yaw_ref_eff = self._wrap_pi(self.yaw_ref + (np.pi if (self._yaw_ref_pi_for_100 or self.yaw_ref_add_pi) else 0.0))
                    err = self._wrap_pi(yaw_m - yaw_ref_eff) * self.imu_err_sign
                    if abs(err) < self.imu_deadband: err = 0.0
                    ctrl = neutral + self.k_yaw_st * err
                else:
                    ctrl = neutral

                ctrl = float(np.clip(ctrl, 0.5 - 0.12, 0.5 + 0.12))
                self.directControl = ctrl
                self.speed = self.second_straight_speed

                if self.straight_imu_on and (now - self.straight_imu_start) >= self.straight_imu_window:
                    self.straight_imu_on = False
                    rospy.loginfo("[STRAIGHT IMU] window END")

                if (now - self.straight2_start) >= self.second_straight_duration:
                    self.directControl = None
                    self.sequence = -1
                    self.speed = 1800
                    self.straight2_start = None
                    self.straight_imu_on = False
                    self.pidcal.reset()
                    self.pidcal._d_freeze_until = rospy.get_time() + 0.15
                    rospy.loginfo("[STRAIGHT] handoff → PID")
                    return

            elif self.sequence == 200:
                rospy.loginfo("[FINAL STOP] Shutting down node (steer neutral, no stop).")
                self._publish_shutdown_steer_then_exit()
                return

    # ==== 패치 B: 빠른 디바운스 stop_line ====
    def stop_line(self, lane=None):
        if self.stop_line_count >= 6:
            if self.sequence in (90, 91, 100, 200) or self.exiting:
                return
            nowt = rospy.get_time()

            # TURN 후 무시 윈도우
            if nowt < getattr(self, '_ignore_stopline_until', 0.0):
                self._stop_det_frames = 0
                return

            if lane is None:
                lane = self.img
            H, W = lane.shape[:2]

            # 핸드오프 직후 2초는 더 민감하게(옵션)
            aggr = (nowt - getattr(self, "_handoff_ts", 0.0) < 2.0)
            band = (0.88, 0.99) if aggr else self.STOP_BAND
            min_frac = 0.10 if aggr else self.STOP_MIN_FRAC

            band_lo = int(band[0] * H)
            band_hi = int(band[1] * H)

            min_white_px = int(min_frac * W)
            thr = 255 * min_white_px

            r_hist = np.sum(lane, axis=1).astype(np.int64)
            if band_lo > 0:
                r_hist[:band_lo] = 0
            if band_hi < H:
                r_hist[band_hi:] = 0
            r_hist[r_hist < thr] = 0

            if r_hist.sum() <= 0:
                self._stop_det_frames = 0
                return

            # 검출됨 → 연속 프레임 카운트
            self._stop_det_frames += 1

            if (self._stop_det_frames >= self.STOP_REQ_FRAMES) and ((nowt - self._last_stop_ts) > self.STOP_COOLDOWN):
                self._last_stop_ts = nowt
                self._stop_det_frames = 0

                self.stopline_count += 1
                rospy.loginfo(f"[STOPLINE] hit #{self.stopline_count}")

                # ==== 4개로 늘어난 매핑 반영 ====
                action = self.STOP_ACTIONS.get(self.stopline_count, "none")

                if action == "right_turn":
                    # 1번째: 우회전
                    self.sequence = 91
                    self.turn_delay_start = time.time()

                elif action == "straight_imu":
                    # 2번째: 직진(IMU)
                    self.sequence = 100

                elif action == "exit":
                    # 4번째: 종료
                    rospy.loginfo("[FINAL STOP] Fourth stop line reached. Exit without stopping (steer neutral).")
                    self._publish_shutdown_steer_then_exit()
                    return

                else:
                    # 3번째: 아무 동작 없음(통과)
                    rospy.loginfo("[STOPLINE] mapped action: NONE (continue)")

    # ================= TURN FSM =================
    def leftturn_fsm(self):
        if self.stop_line_count >= 6:
            if self.mode == "WAIT":
                self.speed = 0
                self._set_steer_center()

                allow_left = (self.signal in (16, 33))
                if allow_left and not self._allow_left_prev:
                    self.mode = "WAIT_HOLD"
                    self.wait_hold_start = time.monotonic()
                    rospy.loginfo("[FSM] WAIT -> WAIT_HOLD  (arm left-turn)")
                self._allow_left_prev = allow_left

            elif self.mode == "WAIT_HOLD":
                self.speed = 0
                self._set_steer_center()
                if time.monotonic() - self.wait_hold_start >= self.WAIT_HOLD_TIME:
                    self.mode = "TURN"
                    self.turn_start = time.monotonic()
                    rospy.loginfo("[FSM] WAIT_HOLD → TURN (force-left)")

            elif self.mode == "TURN":
                elapsed = time.monotonic() - self.turn_start
                left_val = (1.0 - self.STEER_LEFT) if self.INVERT_STEER else self.STEER_LEFT

                if self.SLEW_BYPASS_IN_TURN:
                    self.prev_steer = left_val
                    self._publish_steer(left_val)  # 즉시 적용
                else:
                    self._set_steer_slew(left_val)

                self.speed = 0 if elapsed < self.TURN_PRE_HOLD else self.TURN_SPEED
                self._publish_speed(self.speed)

                if elapsed >= self.TURN_TIME:
                    self.mode = "HANDOFF"
                    # PID 안정화용 약간의 D freeze
                    self.pidcal.reset()
                    self.pidcal._d_freeze_until = rospy.get_time() + 0.10
                    # ==== 패치 A: 무시시간 0.2s로 축소 + 핸드오프 시각 기록 ====
                    now = rospy.get_time()
                    self._ignore_stopline_until = now + 0.2
                    self._handoff_ts = now
                    rospy.loginfo("[FSM] TURN → HANDOFF (stopline ignore 0.2s)")

            elif self.mode == "HANDOFF":
                # 다음 루프부터 FOLLOW 파이프라인으로
                self.mode = "FOLLOW"

    # ================= Helpers (publish/steer) =================
    def _publish_speed(self, v):
        if self.stop_line_count >= 6:
            self.speed_msg.data = float(v)
            self.speed_pub.publish(self.speed_msg)

    def _publish_steer(self, s_norm):
        if self.stop_line_count >= 6:
            s_norm = float(np.clip(s_norm, 0.0, 1.0))
            self.steer_msg.data = s_norm
            self.steer_pub.publish(self.steer_msg)

    def _set_steer_center(self):
        if self.stop_line_count >= 6:
            self.prev_steer = self.STEER_CENTER
            self._publish_steer(self.STEER_CENTER)

    def _set_steer_slew(self, val):
        if self.stop_line_count >= 6:
            val = float(np.clip(val, 0.0, 1.0))
            delta = val - self.prev_steer
            lim = self.SLEW_UP if delta >= 0 else self.SLEW_DOWN
            if abs(delta) > lim:
                val = self.prev_steer + np.sign(delta) * lim
            self.prev_steer = val
            self._publish_steer(val)

    # ================= Shutdown utils =================
    def _publish_shutdown_steer_then_exit(self):
        if self.stop_line_count >= 6:
            self.exiting = True
            try:
                for _ in range(int(self.shutdown_publish_repeats)):
                    self._pub_all(self.shutdown_steer_value, float(self.speed))
                    rospy.sleep(float(self.shutdown_publish_dt))
                rospy.sleep(0.2)
            except Exception:
                pass
            rospy.signal_shutdown("Fourth stop line — exit without stopping")

    def _on_shutdown(self):
        if self.stop_line_count >= 6:
            try:
                for _ in range(int(self.shutdown_publish_repeats)):
                    self._pub_all(self.shutdown_steer_value, float(self.speed))
                    rospy.sleep(float(self.shutdown_publish_dt))
                rospy.loginfo(f"[SHUTDOWN] steer set to {self.shutdown_steer_value} (repeated)")
            except Exception:
                pass

    def _wrap_pi(self, a):
        if self.stop_line_count >= 6:
            return (a + np.pi) % (2 * np.pi) - np.pi

    # ================= Main loop =================
    def run(self):
        while not rospy.is_shutdown():
            if self.slam_status == True:
                if self.stop_line_count >= 6:
                    if not self.boot_armed:
                        if self.img_ready and self.tl_ready and (time.monotonic() - self.boot_start) > self.MIN_BOOT_HOLD:
                            self.boot_armed = True
                        else:
                            self.prev_steer = self.STEER_CENTER
                            self._pub_all(self.prev_steer, 0.0)
                            rospy.sleep(1.0/30.0)
                            continue
                    if self.cv_img is None:
                        rospy.sleep(0.01)
                        continue

                    # 1) 신호등 좌회전 FSM 단계: 인지 스킵, 고정 스크립트
                    if self.mode in ("WAIT", "WAIT_HOLD", "TURN", "HANDOFF"):
                        self.leftturn_fsm()
                        # TURN/WAIT 단계는 여기서 바로 Publish됨
                        if self.mode != "FOLLOW":
                            rospy.sleep(1.0/30.0)
                            continue

                    # 2) FOLLOW 파이프라인 (LaneFollower 흐름)
                    # 인지
                    self.img_init(self.cv_img)
                    self.img_transform()

                    # (A) 노란선 warp
                    self.img_warp(self.yellow_range, change_img=False, warp_img_zoomx=self.x // 2.5)
                    self.yellow_warped = self.warped_img

                    # (B) 흰색 warp: 정지선 검출 전용
                    self.img_warp(self.white_range, change_img=False, warp_img_zoomx=self.x // 2.5)
                    self.white_warped = self.warped_img

                    # (C) 합성마스크 warp (표시/백업)
                    self.img_warp()  # self.combined_range -> self.img

                    # 판단/시퀀스
                    self.go_sequence()

                    # 정지선은 흰색 워프 사용 (빠른 디바운스)
                    self.stop_line(lane=self.white_warped)

                    # 제어
                    if self.directControl is not None:
                        self._pub_all(self.directControl, float(self.speed))
                    else:
                        self.control_pub(ctrl=None)

            rospy.sleep(1.0/30.0)

# ===================== MAIN =====================
if __name__ == "__main__":
    node = IntegratedLeftTurnAndLane()
    node.run()
