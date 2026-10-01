#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2
from cv2 import cvtColor, COLOR_BGR2HSV
from cv_bridge import CvBridge

class PidCal:
    # 필요 시 외부에서 직접 사용 가능(현재 process는 PID를 직접 쓰지 않음)
    error_sum = 0.0
    error_old = 0.0
    # 기본 PID 게인 (원래 사용값 보존 가능)
    p = [0.0028, 0.0000000000001, 0.0065]

    def __init__(self):
        self.x = 0.0

    def pid_control(self, x_current, setpoint=320.0):
        self.x = x_current
        error = setpoint - x_current
        p1 = self.p[0] * error
        self.error_sum += error
        i1 = self.p[1] * self.error_sum
        d1 = self.p[2] * (error - self.error_old)
        self.error_old = error
        pid = p1 + i1 + d1
        return pid

class LaneFollower:
    def __init__(self) -> None:
        # 외부 입력 이미지 버퍼
        self.cv_img = None
        self.bridge = CvBridge()

        # Perception buffers
        self.img = None
        self.img_hsv = None
        self.y, self.x = None, None
        self.yellow_range = None
        self.white_range = None
        self.combined_range = None

        # Warp
        self.warp_img_size = None
        self.warp_img_zoomx = None
        self.warped_img = None

        # Sliding window
        self.out_img = None
        self.nwindows = 12
        self.margin = 60
        self.minpix = 5
        self.threshold = 100
        self.l_lane, self.r_lane = None, None

        # 중앙 유지용 pos
        self.pos = None
        self.alpha = 0.8          # EMA 필터 계수
        self.jump_thresh = 150.0  # 프레임간 급변 억제
        self.pos_min = 0.0
        self.pos_max = 1e9

    # ========== 외부에서 호출되는 API ==========
    def process(self, bgr_img):
        """
        rotary.py(또는 rotary_sum.py) 호환 인터페이스
        입력: BGR 이미지(numpy)
        출력: (steer, pos_norm)
          - steer: 0.0(좌) ~ 1.0(우), 0.5 직진
          - pos_norm: 차선 중앙의 0~1 정규화 위치
        """
        if bgr_img is None or not hasattr(bgr_img, "shape"):
            return 0.5, 0.5

        # 1) 입력/전처리
        self.img_init(bgr_img)
        self.img_transform()

        # 2) 워프: 노란선 기준 보조 워프 시도 후 메인 워프
        try:
            self.img_warp(self.yellow_range, change_img=False, warp_img_zoomx=self.x // 2.5)
        except Exception:
            pass
        self.img_warp()

        # 3) 슬라이딩 윈도우로 차선 검출
        self.sliding_window()

        # 4) 중앙 위치 계산 + 스무딩
        center_pos_raw = self.compute_center_pos()
        if center_pos_raw is None or self.x is None or self.x <= 0:
            return 0.5, 0.5

        self.pos = self._smooth_pos(self.pos, center_pos_raw)

        # 5) 정규화 및 조향값 산출
        pos_norm = float(np.clip(self.pos / float(self.x), 0.0, 1.0))
        steer = float(np.clip(pos_norm, 0.0, 1.0))  # 화면 중앙(0.5) 기준
        return steer, pos_norm

    # ========== 내부 유틸리티 ==========
    def img_init(self, img):
        self.img = img
        self.y, self.x, _ = img.shape
        # HSV 변환
        self.img_hsv = cvtColor(img, COLOR_BGR2HSV)
        # Warp 파라미터
        self.warp_img_size = [self.x, self.x]
        self.warp_img_zoomx = self.x // 4

    def img_transform(self):
        img_hsv = self.img_hsv
        # 노란색 범위
        # lower_yellow = np.array([15, 100, 140], dtype=np.uint8)
        # upper_yellow = np.array([30, 200, 255], dtype=np.uint8)
        # self.yellow_range = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        # 흰색 범위
        lower_white = np.array([0, 0, 140], dtype=np.uint8)
        upper_white = np.array([50, 70, 255], dtype=np.uint8)
        self.white_range = cv2.inRange(img_hsv, lower_white, upper_white)
        # 합치기
        # self.combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)
        self.img = self.white_range

    def img_warp(self, img=None, change_img=True, warp_img_zoomx=None):
        y, x = self.y, self.x
        if img is None:
            img = self.img
        warp_img_size = self.warp_img_size
        if warp_img_zoomx is None:
            warp_img_zoomx = self.warp_img_zoomx

        # source points (기존 사용 버전 유지)
        # topx = 269
        # bottomx = -25
        # topy = 271
        # bottomy = y
        src_points = np.float32([
            [0,450],
            [285,260],
            [x-285,260],
            [x,450]
        ])

        # destination points
        topx = warp_img_zoomx
        bottomx = warp_img_zoomx
        topy = x // 4
        bottomy = x
        dst_points = np.float32([
            [bottomx, bottomy],
            [topx, topy],
            [x - topx, topy],
            [x - bottomx, bottomy]
        ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.warped_img = cv2.warpPerspective(img, matrix, warp_img_size)

        if change_img:
            self.img = self.warped_img
            self.x, self.y = warp_img_size

    def sliding_window(self):
        nwindows = self.nwindows
        margin = self.margin
        minpix = self.minpix
        lane = self.img
        midpoint = self.x // 2
        threshold = self.threshold

        histogram = np.sum(lane, axis=0)
        leftx_current = int(np.argmax(histogram[:midpoint])) if midpoint > 0 else 0
        rightx_current = int(np.argmax(histogram[midpoint:]) + midpoint) if midpoint < len(histogram) else 0

        window_height = max(1, int(self.x / max(1, nwindows)))
        nz = lane.nonzero()  # nz[0]: y indices, nz[1]: x indices

        left_lane_inds = []
        right_lane_inds = []

        lx, ly, rx, ry = [], [], [], []
        try:
            self.out_img = np.dstack((lane, lane, lane)) * 255
        except Exception:
            self.out_img = None

        l_err = [0, 0]
        r_err = [0, 0]
        foundr, foundl = (False, False)

        for window in range(nwindows):
            win_yl = self.x - (window + 1) * window_height
            win_yh = self.x - window * window_height

            win_xll = leftx_current - margin
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            # 올바른 마스크: y는 nz[0], x는 nz[1]
            good_left_mask = (nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xll) & (nz[1] < win_xlh)
            good_right_mask = (nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)

            good_left_inds = np.where(good_left_mask)[0]
            good_right_inds = np.where(good_right_mask)[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                if threshold < leftx_current < (len(histogram) - threshold):
                    l_err[1] = 0
                    foundl = True
                else:
                    l_err[1] += 1
                leftx_current = int(np.mean(nz[1][good_left_inds]))
            else:
                l_err[1] += 1
            if not foundl:
                l_err[0] += 1

            if len(good_right_inds) > minpix:
                if threshold < rightx_current < (len(histogram) - threshold):
                    r_err[1] = 0
                    foundr = True
                else:
                    r_err[1] += 1
                rightx_current = int(np.mean(nz[1][good_right_inds]))
            else:
                r_err[1] += 1
            if not foundr:
                r_err[0] += 1

            lx.append(leftx_current)
            ly.append((win_yl + win_yh) / 2.0)
            rx.append(rightx_current)
            ry.append((win_yl + win_yh) / 2.0)

        try:
            left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
            right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)
            if self.out_img is not None:
                if left_lane_inds.size:
                    self.out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
                if right_lane_inds.size:
                    self.out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]

        except Exception:
            pass

        self.l_lane, self.r_lane = (lx, ly, l_err), (rx, ry, r_err)

    def compute_center_pos(self):
        if not self.l_lane or not self.r_lane:
            return None
        try:
            posl = int(self.l_lane[0][3])  # x 리스트에서 4번째 요소
            posr = int(self.r_lane[0][3])
        except Exception:
            return None
        pos = (posl + posr) // 2
        return float(pos)

    def _smooth_pos(self, prev_pos, new_pos):
        if prev_pos is None:
            return float(new_pos)
        if abs(new_pos - prev_pos) > self.jump_thresh:
            new_pos = prev_pos
        pos = self.alpha * new_pos + (1.0 - self.alpha) * prev_pos
        pos = max(self.pos_min, min(self.pos_max, pos))
        return pos
