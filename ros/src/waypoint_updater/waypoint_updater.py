#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

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

LOOKAHEAD_WPS = 30 #200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.current_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
	# the current position: x,y
	self.cur_pos = None
	# the base waypoints
	self.base_waypoints = None
	# the closest waypoint index
	self.closest_waypoint_index = 0
	
        rospy.spin()
	
    # help function for calculating distance between 2 waypoints
    def calc_distance(self, wp1, wp2):
	return math.sqrt((wp1.x-wp2.x)**2 + (wp1.y-wp2.y)**2 + (wp1.z-wp2.z)**2)
	

    def pose_cb(self, msg):
        # TODO: Implement
        
	# current position
	self.cur_pos = msg.pose.position
	#cur_x = msg.pose.position.x
	#cur_y = msg.pose.position.y
	#rospy.loginfo("--------The current position: %f,%f",cur_x,cur_y)
	
	# in case the base_waypoints are not loaded yet
	if not self.base_waypoints:
		return	
	
	# Find the waypoint index where the waypoint is closest to the current position
	# Initialize a distance value
	closest_dist = None
	for i in range(len(self.base_waypoints)):
		wpt = self.base_waypoints[i].pose.pose.position
		dist = self.calc_distance(wpt, self.cur_pos)
		if (closest_dist is None) or (dist < closest_dist):
			closest_dist = dist
			self.closest_waypoint_index = i
			
	# Print out the result
	rospy.loginfo("The Closest Waypoing index: {}, the distance is: {}.".format(i,closest_dist))
	
	# Create lookahead waypoints from the closest waypoint
	lane = Lane()  # create a msg type
	lane.waypoints = self.base_waypoints[self.closest_waypoint_index: self.closest_waypoint_index+ LOOKAHEAD_WPS]
	
	
	'''for i in range(self.closest_waypoint_index, self.closest_waypoint_index + LOOKAHEAD_WPS):
		index = i % len(self.base_waypoints)
		lane.waypoints.append(wpt)
	'''
	
	self.final_waypoints_pub.publish(lane)
	
    def waypoints_cb(self, waypoints):
        # TODO: Implement
	# Load the base waypoints once
	if not self.base_waypoints:
		self.base_waypoints = waypoints.waypoints
	# Unsubsribe from the topic
	self.base_waypoints_sub.unregister(self)        

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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
