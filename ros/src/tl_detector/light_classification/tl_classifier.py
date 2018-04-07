from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
import time


RED = TrafficLight.RED
GREEN = TrafficLight.GREEN
YELLOW = TrafficLight.YELLOW
UNKNOWN = TrafficLight.UNKNOWN

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        
        # get the real path dir, otherwise it won't find the pb file
        pwd = os.path.dirname(os.path.realpath(__file__))
        PATH_TO_MODEL = os.path.join(pwd,'frozen_inference_graph2.pb')
        PATH_TO_LABELS = os.path.join(pwd,'sim_label_map.pbtxt')
        NUM_CLASSES = 3
		
        self.image_np = None
        
        # Loading the lable map
        self.category_index = {1: {'id':1, 'name': u'red'}, 2: {'id':2, 'name': u'yellow'}, 3: {'id':3, 'name': u'green'} }
        
        # load a FROZEN TF model into memory        
        self.detection_graph = tf.Graph()
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph) #, config=config
        #self.sess = tf.Session(graph=self.detection_graph, config=config)



    def get_classification(self, img):
        """Determines the color of the traffic light in the image
        
        Args:
            image (cv::Mat): image containing the traffic light
        
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        
        """
        #TODO implement light color prediction
        
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)
            
            min_score_thresh = .5            
            current_light = UNKNOWN
            
            for i in range(boxes.shape[0]):
                if scores is None or scores[i] > min_score_thresh:
                    class_name = self.category_index[classes[i]]['name']
                    # class_id = self.category_index[classes[i]]['id']
                    #print(class_name)
                    
                    if class_name == 'red':
                        current_light = RED
                    elif class_name == 'green':
                        current_light = GREEN
                    elif class_name == 'yellow':
                        current_light = YELLOW                              
            
            
        return current_light
