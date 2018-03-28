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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

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

    def pose_cb(self, msg):
        # TODO: Implement
        
	# current position
	self.cur_pos = msg.pose.position
	#cur_x = msg.pose.position.x
	#cur_y = msg.pose.position.y
	#rospy.loginfo("--------The current position: %f,%f",cur_x,cur_y)
	
	
    def waypoints_cb(self, waypoints):
        # TODO: Implement
	# Load the base waypoints once
	if not self.base_waypoints:
		self.base_waypoints = waypoints.waypoints
	# Initialize a distance value
	closest_dist = 1000000
	# the first waypoint which is closest to the current position
	for i in range(len(self.base_waypoints)):
		wpt = self.base_waypoints[i].pose.pose.position
		distance = math.sqrt((wpt.x-self.cur_pos.x)**2 + (wpt.y-self.cur_pos.y)**2 + (wpt.z-self.cur_pos.z)**2)
		if distance < closest_dist:
			closest_dist = distance
			self.closest_waypoint_index = i
	# Create lookahead waypoints from the closest waypoint
	for i in range(self.closest_waypoint_index, self.closest_waypoint_index + LOOKAHEAD_WPS):
		index = i % len(self.base_waypoints)
		lane.waypoints.append(wpt)
	#lane = Lane()
	#lane.waypoints = msg.position.x
	#self.final_waypoints_pub.publish(lane)

        pass

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
