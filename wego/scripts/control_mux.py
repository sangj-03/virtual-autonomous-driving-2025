#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os, sys
import rospy
from std_msgs.msg import Float64, Int32
from ackermann_msgs.msg import AckermannDriveStamped

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)
from stop_line_detect import StopLineTest

class ControlMux:
    def __init__(self):
        rospy.init_node("control_mux")

        # Subscriber
        rospy.Subscriber("/go_to_rotary/speed_cmd", Float64, self.speed_gtr_CB, queue_size=1)
        rospy.Subscriber("/go_to_rotary/steer_cmd", Float64, self.steer_gtr_CB, queue_size=1)

        rospy.Subscriber("/rotary/speed_cmd", Float64, self.speed_rot_CB, queue_size=1)
        rospy.Subscriber("/rotary/steer_cmd", Float64, self.steer_rot_CB, queue_size=1)

        rospy.Subscriber("/ttfm/speed_cmd", Float64, self.speed_ttfm_CB, queue_size=1)
        rospy.Subscriber("/ttfm/steer_cmd", Float64, self.steer_ttfm_CB, queue_size=1)

        # rospy.Subscriber("/traffic/speed_cmd", Float64, self.speed_trf_CB, queue_size=1)
        # rospy.Subscriber("/traffic/steer_cmd", Float64, self.steer_trf_CB, queue_size=1)
# 
        # rospy.Subscriber("/five_mile/speed_cmd", Float64, self.speed_fiv_CB, queue_size=1)
        # rospy.Subscriber("/five_mile/steer_cmd", Float64, self.steer_fiv_CB, queue_size=1)

        rospy.Subscriber("/navigation/status", Int32, self.slam_status_CB, queue_size=1)

        # Publisher
        self.ack_pub = rospy.Publisher("high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=10)
        self.count_pub = rospy.Publisher("/stop_line/count", Int32, queue_size=1)
        self.speed_gtr = 0.0
        self.steer_gtr = 0.5
        self.speed_rot = 0.0
        self.steer_rot = 0.5
        self.speed_ttfm = 0.0
        self.steer_ttfm = 0.5
        # self.speed_trf = 0.0
        # self.steer_trf = 0.5
        # self.speed_fiv = 0.0
        # self.steer_fiv = 0.5

        self.sld = StopLineTest()        
        self.select = 0     # 0: go_to_rotary, 1: rotary, 2: traffic
        self.speed_msg = None
        self.steer_msg = None
        self.slam_status = False
        self.last_select = None

        self.rate = rospy.Rate(60)
        rospy.loginfo("!! control_mux ready. !!")

    def slam_status_CB(self, msg):
        self.slam_status = msg.data

    def mux_stop_line(self):
        self.sld.stop_line_pub()
        self.count_pub.publish(Int32(data=self.sld.stop_line_count))

    def select_CB(self):
        if self.sld.stop_line_count < 5:
            self.select = 0     # go_to_rotary
        elif 5 <= self.sld.stop_line_count < 6:
            self.select = 1     # roatry
        elif 6 <= self.sld.stop_line_count:
            self.select = 2     # traffic_to_five_mile

    def speed_gtr_CB(self, msg): self.speed_gtr = msg.data
    def steer_gtr_CB(self, msg): self.steer_gtr = msg.data

    def speed_rot_CB(self, msg): self.speed_rot = msg.data
    def steer_rot_CB(self, msg): self.steer_rot = msg.data

    def speed_ttfm_CB(self, msg): self.speed_ttfm = msg.data
    def steer_ttfm_CB(self, msg): self.steer_ttfm = msg.data

    # def speed_trf_CB(self, msg): self.speed_trf = msg.data
    # def steer_trf_CB(self, msg): self.steer_trf = msg.data
# 
    # def speed_fiv_CB(self, msg): self.speed_fiv = msg.data
    # def steer_fiv_CB(self, msg): self.steer_fiv = msg.data

    def commands_to_ackermann(self, motor_speed, servo_pos):
        speed_per_unit = (3.2 * 1000.0 / 3600.0) / 1000.0  # 0.888888... / 1000
        ack_speed = motor_speed * speed_per_unit

        if servo_pos is None:
            servo_pos = 0.5

        servo_pos = max(0.0, min(1.0, float(servo_pos)))
        ack_steer = (0.5 - servo_pos) * (0.34 / 0.5)

        return ack_speed, ack_steer

    def run(self):
        while not rospy.is_shutdown():
            self.mux_stop_line()
            self.select_CB()

            if self.slam_status == True:
               # print(" SLAM 종료!! 이제 select 조건문 안으로 들어옴!!!!!")
                if self.select == 0:
                    self.speed_msg = self.speed_gtr
                    self.steer_msg = self.steer_gtr
                elif self.select == 1:
                    self.speed_msg = self.speed_rot
                    self.steer_msg = self.steer_rot
                elif self.select == 2:
                    self.speed_msg = self.speed_ttfm
                    self.steer_msg = self.steer_ttfm

                if self.select != self.last_select:
                    if self.select == 0:
                        print(f"[[ control_mux ]]   go_to_rotary 제어권 획득")
                    elif self.select == 1:
                        print(f"[[ control_mux ]]   rotary 제어권 획득")
                    elif self.select == 2:
                        print(f"[[ control_mux ]]   ttfm 제어권 획득")
                self.last_select = self.select

                # commands -> ackermann convert
                ack_speed, ack_steer = self.commands_to_ackermann(self.speed_msg, self.steer_msg)
                ack = AckermannDriveStamped()
                ack.header.stamp = rospy.Time.now()
                ack.drive.speed = ack_speed
                ack.drive.steering_angle = ack_steer

                self.ack_pub.publish(ack)
            self.rate.sleep()

def main():
    mux = ControlMux()
    while not rospy.is_shutdown():
        mux.run()

if __name__=="__main__":
    main()
