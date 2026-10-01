import rospy
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
import actionlib
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32

class NavigationClient():
    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.current_position=None
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_cb)
        self.shutdown_requested = False 

        self.change = rospy.Publisher("/navigation/status", Int32, queue_size=1)
        # self.status = False
        self.goal_list = []

        # # waypoint 1
        waypoint_1 = MoveBaseGoal()
        waypoint_1.target_pose.header.frame_id = "map"
        waypoint_1.target_pose.pose.position.x = 8.497999954223633
        waypoint_1.target_pose.pose.position.y = -9.580999851226807
        waypoint_1.target_pose.pose.position.z = -0.03
        waypoint_1.target_pose.pose.orientation.w = 0.9978111044642102
        waypoint_1.target_pose.pose.orientation.z = 0.06612866101708949
        self.goal_list.append(waypoint_1)

        # waypoint 3
        waypoint_2 = MoveBaseGoal()
        waypoint_2.target_pose.header.frame_id = "map"
        waypoint_2.target_pose.pose.position.x = 14.235000944137573
        waypoint_2.target_pose.pose.position.y = 0.00900074005127
        waypoint_2.target_pose.pose.position.z = -0.029555841088295
        waypoint_2.target_pose.pose.orientation.w = 0.9978111044642102
        waypoint_2.target_pose.pose.orientation.z = 0.06612866101708949
        self.goal_list.append(waypoint_2)

        waypoint_3 = MoveBaseGoal()
        waypoint_3.target_pose.header.frame_id = "map"
        waypoint_3.target_pose.pose.position.x = 3.350000905990601
        waypoint_3.target_pose.pose.position.y = -7.7820252895355225
        waypoint_3.target_pose.pose.position.z = -0.0295561242103577
        waypoint_3.target_pose.pose.orientation.w = 0.9978111044642102
        waypoint_3.target_pose.pose.orientation.z = 0.06612866101708949
        self.goal_list.append(waypoint_3)

        waypoint_4 = MoveBaseGoal()
        waypoint_4.target_pose.header.frame_id = "map"
        waypoint_4.target_pose.pose.position.x =  18.301542942942007
        waypoint_4.target_pose.pose.position.y = -10.165300329485445
        waypoint_4.target_pose.pose.position.z = -0.004585849141274627
        waypoint_4.target_pose.pose.orientation.w = 0.9999894849385434
        self.goal_list.append(waypoint_4)
        self.current_goal_index = 0
        self.goal_sent = False
        self.waiting_to_send_next = False
        self.wait_start_time = None

    def odom_cb(self, msg):
        self.current_position = msg.pose.pose.position
         # 2) ROS 로그로 깔끔하게 (버퍼/스레드 안전)

    def send_current_goal(self):
        goal = self.goal_list[self.current_goal_index]
        goal.target_pose.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.loginfo(f"Goal {self.current_goal_index + 1} sent.")
        self.goal_sent = True

    def run(self):
        state = self.client.get_state()

        if not self.goal_sent:
            self.send_current_goal()

        elif state == GoalStatus.SUCCEEDED:
           if not self.waiting_to_send_next:
              self.waiting_to_send_next = True
              self.wait_start_time = rospy.Time.now()
              rospy.loginfo("Goal reached. Waiting 2.1 seconds before next...")

           elif (rospy.Time.now() - self.wait_start_time).to_sec() > 1.5:
              if self.current_goal_index == len(self.goal_list) - 1:
                rospy.loginfo("🎉 All goals completed. Shutting down node.")
                self.shutdown_requested = True   # 🚫 signal_shutdown() 삭제
              else:
                self.current_goal_index += 1
                self.goal_sent = False
                self.waiting_to_send_next = False

        elif state in [GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED]:
            rospy.logwarn("Goal failed. Retrying...")
            self.goal_sent = False
        if self.current_position:
            goal = self.goal_list[self.current_goal_index].target_pose.pose.position
            dist = math.hypot(goal.x - self.current_position.x,
                              goal.y - self.current_position.y)
            # rospy.loginfo_throttle(1.0, f"🚗 현재 목표와 거리: {dist:.3f}m")

    def stop(self):
        self.client.cancel_all_goals()
        rospy.loginfo("All goals cancelled.")

def main():
    rospy.init_node("navigation_client")
    nc = NavigationClient()
    rate = rospy.Rate(8)

    while not rospy.is_shutdown():
        nc.run()
        nc.change.publish(nc.shutdown_requested)
        if nc.shutdown_requested:
            break  # 자연스럽게 while 루프 탈출
        rate.sleep()

if __name__ == "__main__":
    main()
