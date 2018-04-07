

Utils help function files error:

```ImportError: No module named utils ```

solution:  I had to get rid of files in this folder, wrote the code directly in the tl_classifier.py.

NotFound the .pb file error:

```
core service [/rosout] found
process[styx_server-1]: started with pid [2406]
process[unity_simulator-2]: started with pid [2407]
process[dbw_node-3]: started with pid [2412]
process[waypoint_loader-4]: started with pid [2413]
[unity_simulator-2] process has finished cleanly
log file: /home/student/.ros/log/b68ef742-3a5c-11e8-bd21-080027345f02/unity_simulator-2*.log
process[pure_pursuit-5]: started with pid [2414]
process[waypoint_updater-6]: started with pid [2417]
process[tl_detector-7]: started with pid [2450]
(2406) wsgi starting up on http://0.0.0.0:4567
Traceback (most recent call last):
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/tl_detector.py", line 181, in <module>
    TLDetector()
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/tl_detector.py", line 46, in __init__
    self.light_classifier = TLClassifier()
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/light_classification/tl_classifier.py", line 37, in __init__
    serialized_graph = fid.read()
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/lib/io/file_io.py", line 118, in read
    self._preread_check()
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/lib/io/file_io.py", line 78, in _preread_check
    compat.as_bytes(self.__name), 1024 * 512, status)
  File "/usr/lib/python2.7/contextlib.py", line 24, in __exit__
    self.gen.next()
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/framework/errors_impl.py", line 466, in raise_exception_on_not_ok_status
    pywrap_tensorflow.TF_GetCode(status))
tensorflow.python.framework.errors_impl.NotFoundError: frozen_inference_graph2.pb
```


tf version error:

```
[ERROR] [1523107062.921475]: bad callback: <bound method TLDetector.image_cb of <__main__.TLDetector object at 0x7fa208d45ed0>>
Traceback (most recent call last):
  File "/opt/ros/kinetic/lib/python2.7/dist-packages/rospy/topics.py", line 750, in _invoke_callback
    cb(msg)
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/tl_detector.py", line 80, in image_cb
    light_wp, state = self.process_traffic_lights()
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/tl_detector.py", line 172, in process_traffic_lights
    state = self.get_light_state(light)
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/tl_detector.py", line 136, in get_light_state
    return self.light_classifier.get_classification(cv_image)
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/light_classification/tl_classifier.py", line 71, in get_classification
    feed_dict={self.image_tensor: img_expanded})
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 895, in run
    run_metadata_ptr)
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1124, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1321, in _do_run
    options, run_metadata)
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1340, in _do_call
    raise type(e)(node_def, op, message)
InvalidArgumentError: NodeDef mentions attr 'identical_element_shapes' not in Op<name=TensorArrayV3; signature=size:int32 -> handle:resource, flow:float; attr=dtype:type; attr=element_shape:shape,default=<unknown>; attr=dynamic_size:bool,default=false; attr=clear_after_read:bool,default=true; attr=tensor_array_name:string,default=""; is_stateful=true>; NodeDef: Preprocessor/map/TensorArray = TensorArrayV3[clear_after_read=true, dtype=DT_FLOAT, dynamic_size=false, element_shape=<unknown>, identical_element_shapes=true, tensor_array_name="", _device="/job:localhost/replica:0/task:0/cpu:0"](Preprocessor/map/strided_slice). (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
         [[Node: Preprocessor/map/TensorArray = TensorArrayV3[clear_after_read=true, dtype=DT_FLOAT, dynamic_size=false, element_shape=<unknown>, identical_element_shapes=true, tensor_array_name="", _device="/job:localhost/replica:0/task:0/cpu:0"](Preprocessor/map/strided_slice)]]

Caused by op u'Preprocessor/map/TensorArray', defined at:
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/tl_detector.py", line 184, in <module>
    TLDetector()
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/tl_detector.py", line 48, in __init__
    self.light_classifier = TLClassifier()
  File "/home/student/carnd/CarND-Capstone/ros/src/tl_detector/light_classification/tl_classifier.py", line 43, in __init__
    tf.import_graph_def(od_graph_def, name='')
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/framework/importer.py", line 313, in import_graph_def
    op_def=op_def)
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 2630, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/home/student/.local/lib/python2.7/site-packages/tensorflow/python/framework/ops.py", line 1204, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): NodeDef mentions attr 'identical_element_shapes' not in Op<name=TensorArrayV3; signature=size:int32 -> handle:resource, flow:float; attr=dtype:type; attr=element_shape:shape,default=<unknown>; attr=dynamic_size:bool,default=false; attr=clear_after_read:bool,default=true; attr=tensor_array_name:string,default=""; is_stateful=true>; NodeDef: Preprocessor/map/TensorArray = TensorArrayV3[clear_after_read=true, dtype=DT_FLOAT, dynamic_size=false, element_shape=<unknown>, identical_element_shapes=true, tensor_array_name="", _device="/job:localhost/replica:0/task:0/cpu:0"](Preprocessor/map/strided_slice). (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
         [[Node: Preprocessor/map/TensorArray = TensorArrayV3[clear_after_read=true, dtype=DT_FLOAT, dynamic_size=false, element_shape=<unknown>, identical_element_shapes=true, tensor_array_name="", _device="/job:localhost/replica:0/task:0/cpu:0"](Preprocessor/map/strided_slice)]]

```
