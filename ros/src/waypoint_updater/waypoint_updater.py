#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
from scipy.spatial import KDTree
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 30  #50 #70 #30 #200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5
ROS_LOOP_RATE = 5 #10,30,50,70 Hz
PRINT = False #True

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        # the current position: x,y
        self.cur_pos = None
        # the base waypoints
        self.base_waypoints = None
        # KDTree waypoints
        self.kdtree_waypoints = None 
        self.waypoints_2d = None
        # closest waypoint index from the RED light stopline
        self.stop_wp_idx = 0
        
        #if PRINT: rospy.logwarn("MYPRINTING-----START")

        # process some work based on above info
        self.do_process()
	
        #rospy.spin()

    # do something
    def do_process(self):
        rate = rospy.Rate(ROS_LOOP_RATE)
        
        while not rospy.is_shutdown():
            #make sure the current position and waypoints are loaded 
            if self.cur_pos and self.base_waypoints:
                closest_wp_idx = self.get_closest_waypoint_idx()
                lane = self.generate_lane(closest_wp_idx)
                self.final_waypoints_pub.publish(lane)
            
            rate.sleep()

    def get_closest_waypoint_idx(self):
        # get the current car position
        cur_x = self.cur_pos.x
        cur_y = self.cur_pos.y
        
        # use KDTree to find the closest waypoint index
        closest_wp_idx = self.kdtree_waypoints.query([cur_x,cur_y], 1)[1]
                
        # check if the closest waypoint is ahead or behind the car
        closest_coord = self.waypoints_2d[closest_wp_idx]
        prev_coord = self.waypoints_2d[closest_wp_idx-1]
        # hyperplane
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([cur_x, cur_y])                                        	
        
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)                          
        if val > 0:
            closest_wp_idx = (closest_wp_idx + 1) % len(self.waypoints_2d)
        
        #if PRINT: rospy.logwarn("MYPRINTING-----Closest Waypoint Index:{}".format(closest_wp_idx))

        return closest_wp_idx
 
    def generate_lane(self, close_wp_idx):
        
        # the farthest waypoint index
        far_wp_idx = close_wp_idx + LOOKAHEAD_WPS
        # waypoints ahead of the car by which the car follows
        waypoints_ahead = self.base_waypoints.waypoints[close_wp_idx: far_wp_idx]
        
        # Create lookahead waypoints from the closest waypoint              
        lane = Lane()  # create a msg type
        lane.header = self.base_waypoints.header
                                                                        
        # if NOT RED light OR if the end of green waypoint did not reach RED STOP line
        if self.stop_wp_idx == -1 or self.stop_wp_idx >= far_wp_idx:
            lane.waypoints = waypoints_ahead
        else:
            #rospy.logwarn("MYPRINTING-----Stopping at:{}".format(self.stop_wp_idx))
            lane.waypoints = self.decelerate_waypoints(waypoints_ahead, close_wp_idx)
            
            #for i in range(len(waypoints_ahead)):
            #    self.set_waypoint_velocity(waypoints_ahead,i,0)
            #lane.waypoints = waypoints_ahead
        
        return lane
    
        
    def decelerate_waypoints(self, waypoints, close_wp_idx):
        temp = []
        for i,wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            stop_wp_idx = max(self.stop_wp_idx - close_wp_idx -2, 0)
            dist = self.distance(waypoints, i, stop_wp_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            
            #p.twist.twist.linear.x = 0.
            temp.append(p)

        return temp
    
    def pose_cb(self, msg):
        #if PRINT: rospy.logwarn("MYPRINTING-----pose_cb")
        # current position
        self.cur_pos = msg.pose.position
        #rospy.loginfo("MYPRINTING-----The current position: %f,%f",self.cur_pos.x,self.cur_pos.y)

    def waypoints_cb(self, waypoints):
        #if PRINT: rospy.logwarn("MYPRINTING-----waypoints_cb")

        # Load the base waypoints once
        if not self.base_waypoints and not self.kdtree_waypoints:
            self.base_waypoints = waypoints
            # convert the data structure to KDTree So that it is fast to calculate closest waypoint index
            self.waypoints_2d = [[wpt.pose.pose.position.x, wpt.pose.pose.position.y] for wpt in waypoints.waypoints]
            self.kdtree_waypoints = KDTree(self.waypoints_2d)
    
    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. Implement
        #if PRINT: rospy.logwarn("MYPRINTING-----traffic_cb")

        self.stop_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
