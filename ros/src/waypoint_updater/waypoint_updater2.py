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

LOOKAHEAD_WPS = 200 #70 #30 #200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5

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
        # the closest waypoint index
        self.closest_wp_idx = 0
        # the farthest waypoint index
        self.far_wp_idx = 0
        # closest waypoint index from the RED light stopline
        self.stop_wp_idx = 0 
	
        rospy.spin()
	
    # help function for calculating distance between 2 waypoints
    def calc_distance(self, wp1, wp2):
        return math.sqrt((wp1.x-wp2.x)**2 + (wp1.y-wp2.y)**2 + (wp1.z-wp2.z)**2)
	

    def pose_cb(self, msg):
        # TODO: Implement
        
        # current position
        self.cur_pos = msg.pose.position
        cur_x = msg.pose.position.x
        cur_y = msg.pose.position.y
        #rospy.loginfo("MYPRINTING-----The current position: %f,%f",cur_x,cur_y)
	
        # in case the base_waypoints are not loaded yet
        if not self.base_waypoints:
            return	
        
        # use KDTree to find the closest waypoint index
        closest_wp_idx_tmp = self.kdtree_waypoints.query([cur_x,cur_y], 1)[1]
        closest_coord = self.waypoints_2d[closest_wp_idx_tmp]
        prev_coord = self.waypoints_2d[closest_wp_idx_tmp-1]
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([cur_x, cur_y])
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        if val > 0:
            self.closest_wp_idx = (closest_wp_idx_tmp + 1) % len(self.waypoints_2d)

        self.far_wp_idx = self.closest_wp_idx + LOOKAHEAD_WPS
        # waypoints ahead of the car by which the car follows
        waypoints_ahead = self.base_waypoints.waypoints[self.closest_wp_idx: self.far_wp_idx]
        # Print out the result
        #rospy.loginfo("The Closest Waypoing index: {}, the distance is: {}.".format(i,closest_dist))
	
        # Create lookahead waypoints from the closest waypoint
        lane = Lane()  # create a msg type
        lane.header = self.base_waypoints.header

        # if NOT RED light OR if the end of green waypoint did not reach the STOP Line.    
        if self.stop_wp_idx == -1 or self.stop_wp_idx >= self.far_wp_idx:
            lane.waypoints = waypoints_ahead
        else:
            lane.waypoints = self.decelerate_waypoints(waypoints_ahead)
        
        self.final_waypoints_pub.publish(lane)

    def decelerate_waypoints(self, waypoints):
        temp = []
        for i,wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_wp_idx = max(self.stop_wp_idx - self.closest_wp_idx -2, 0)
            dist = self.distance(waypoints, i, stop_wp_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp
	
    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # Load the base waypoints once
        if not self.base_waypoints and not self.kdtree_waypoints:
            self.base_waypoints = waypoints
            # convert the data structure to KDTree So that it is fast to calculate closest waypoint
            self.waypoints_2d = [[wpt.pose.pose.position.x, wpt.pose.pose.position.y] for wpt in waypoints.waypoints]
            self.kdtree_waypoints = KDTree(self.waypoints_2d)
    
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
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
