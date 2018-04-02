# self-driving-car-System-Integration




**Step1: Waypoint Updater Node (partial)**
- waypoint_updater.py
- subscribe the topics: 
- ```/base_waypoints```: all waypoints following the highway in the simulator. We can understand it as a Global map in the real world. As soon as the car follows these waypoints, it won’t go off the road. 
- ```/current_pose```: the car’s current position. 
- then publish on the topic: ```/final_waypoints ``` , publish a fixed number of waypoints ahead of the vehicle with the correct target velocities, depending on traffic lights and obstacles.


_**Notes for coding:**_
- either subscribe or publish topic need to know the exact message type we are dealing with. Otherwise, the error may thrown. For instance, to see what message does ```/current_pose``` has do following in the ROS: ```rostopic type /current_pose```. Grab the msg type and do ``` rosmsg show  geometry_msgs/PoseStamped```. Following info is expected to be shown. 
    ```
    ry_msgs/PoseStamped
    std_msgs/Header header
      uint32 seq
      time stamp
      string frame_id
    geometry_msgs/Pose pose
      geometry_msgs/Point position
        float64 x
        float64 y
        float64 z
      geometry_msgs/Quaternion orientation
        float64 x
        float64 y
        float64 z
        float64 w
    ```
- if the error is thrown: ```ERROR: Cannot load message class for [styx_msgs/Lane]. Are your messages built?```. The problem is probably that  ```source devel/setup.sh``` is did not executed. I had experienced this problem many times in the beginning. 
- to check what has published use ``` rostopic echo /final_waypoints```.
- another option to check if msg are published successfully to the topic: /final_waypoints. If the msg is published successfully, We should see the green waypoints ahead of the car. In the beginning, there was latency. The waypoints are not able to update on time. I removed the roslog.info. It helped a lot.


**Step2: DBW Node**
- In this step, I want to accomplish that the car can follow the green waypoints ahead ignoring the traffic light and obstacles.
In order to let the car move, we need to provide [throttle,brake,steer] to the car. Each one of them is a topic. What we need to do is, publish the corresponding message on to the topic. For example, to provide [throttle] value to the car we need to publish the value, 0.2 (has to be within [0,1]), on the topic [/vehicle/throttle_cmd]. Let’s add following code into the dbw_node.py:
```self.publish(throttle=0.2,brake=0,steer=0)```. We can see the car is start moving now. Great.

- We want the car to be able to switch between automatic and manual driving mode. The topic [/vehicle/dbw_enabled] publish this info. We need to subscribe this topic and decide if we want the car to drive automatically or manually. To check the msg on the topic [/vehicle/dbw_enabled] we can do ``` rostopic echo /vehicle/dbw_enabled```.  We will see ```data: True``` or ```data: False``` by clicking checkbox,manual, on and off in the simulator.

- We need to use the car’s current velocity, waypoint linear velocity and waypoint angular velocity to control the car. The PID controller is adapted in this project. We need to feed Cur_vel, linear_vel, angular_vel to PID and corresponding [throttle,brake,steer] values are generated. I used ```rospy.loginfo``` to print out Cur_vel, linear_vel, angular_vel by subscribing the topics, ```/current_velocity``` and ```/twist_cmd ```.  The printing should look something like below: 
  ```
  [rosout][INFO] 2018-03-30 12:21:53,930: MYPRINTING-----Current Vel: 23.737354608-----
  [rosout][INFO] 2018-03-30 12:21:53,946: MYPRINTING-----Linear, Angular Vel:-6.6062637705,-0.756857457362-----
  [rosout][INFO] 2018-03-30 12:21:53,949: MYPRINTING-----Current Vel: 23.7981386368-----
  [rosout][INFO] 2018-03-30 12:21:53,976: MYPRINTING-----Current Vel: 23.9191925984-----
  [rosout][INFO] 2018-03-30 12:21:53,979: MYPRINTING-----Linear, Angular Vel:-5.18794387709,-0.963772954846-----
  [rosout][INFO] 2018-03-30 12:21:53,987: MYPRINTING-----Current Vel: 23.9191925984-----
  ```
- LowPassFilter:  this technique is adapted in the Controller in order to filter out noise from current velocity, like high frequency velocity.

- one significant problem I encountered was that the green waypoint suppose to be stay in front of the car falls behind the car. I assumed this problem is caused by calculation inefficiency.  In the node, update_waypoints, to calculate the closest waypoint index I implemented a loop that goes through 1000+ waypoints. I need to improve the efficiency of this calculation.  We can change the data structure of the base_waypoints. I adapted KDTree. Finding the closest waypoint index from the DKTree data is much efficient. After implementation, the lookahead waypoints latency issue is solved.

- At the end of this step, we should see:
  - the car is moving in a decent speed.
  - the car is following green waypoints ahead of it.
  - the green waypoints ahead of car are not falling behind anymore.


**Step3: Traffic Light Detection**
- subscribe ``` /vehicle/traffic_lights``` to get the state of lights which are RED, GREEN, YELLOW and UNKNOWN. The corresponding stop line for each traffic light is provided in ``` self.config['stop_line_positions']```.
- loop through each traffic light. If it is RED light, publish the index of waypoint nearest to the traffic light stop line on the topic ```/traffic_waypoint ```. If it is Not RED light, just publish ```-1``` on the ```/traffic_waypoint ```.

_**Note:**_ instead of pulling the state associate with simulator light data, I should implement traffic light classifier model. Detect the traffic light and its state from the topic ```/image_color ```.  Determine if it is RED or Not RED using the model. I will do this in next.

- ```rospy.logwarn()``` is another great tool in ROS for debugging the code. However, remember to turn it off if the code turn out to be correct. These logging did cause latency problem because they consumes computing resources. 
- check the camera option in the simulator on. Otherwise the  callback function [image_cb] won’t run.
- after re-launch the code, use ```rostopic echo /traffic_waypoint``` to check the msg published on the topic. The msg should be look like: ```data: -1``` or ```data: 292```.  The closest waypoint index near the traffic light stop line is shown if the light is RED. Otherwise, it should be shown -1.
- Test after adding above code. Unfortunately, the car won’t stop even it is RED light. Let’s fix it.



**Step4: STOP if it is RED light**
- subscribe the topic ``` /traffic_waypoint``` and determine if it is the red traffic light.
- make sure the car can smoothly slowed to full stop at the traffic light. The smooth means acceleration should not exceed 10 m/s^2 and jerk should not exceed 10 m/s^3 according to what we learned from Path Planning project.
- what we need to do is decelerate waypoint velocities before publishing waypoints to ``` /final_waypoints ```. Updating the value of  ```Twist.pose.position.linear.x``` should slow down the car.


- from the beginning, i decided to use LOOKAHEAD_WPS = 30 because it helps with latency issue.  However, this value seems make the car decelerate in too short time. If I improve it to 200, the latency problem came up again. The green waypoints falls behind of the car, which has caused the car not able to drive along with waypoints. How do I fix this?  
  - I tried LOOKAHEAD_WPS = 30, 70, 100, 200. In the end I decide to use 100. It gives enough space to let the car slow down when it is RED light. And it solves the problem of waypoints falling behind the car.
  - I tried different ROS frequency, ```rospy.rate(Val) where Val=10,30,50,70```.  Using 70 make the problem worth because it requires more frequent calculation. Well 10 did work well, but it is just too slow. What if something happens within this period of time and we want the car to reactive on time. I adapted 30 in the end. Only because my computer can deal with this value well. 


_**Notes:**_
- Adding any function before ```rospy.spin()``` won’t execute. Because it only runs callback function repeatedly.
- No need to ```catkin_make``` when developing node with python. All we need to do after updating the code is ```roslaunch``` again. 
- python coding: define a function with parameter be careful with the parameter orders. I encountered this issue when calling the control function. The parameters are in mismatched order. 
- python coding: defining many variables are cumbersome. You may end up do not know where is the bug. Make sure variables defined with proper name and consistent. 
- why waypoints_cb run only at once?
