
r"""Convert raw PASCAL dataset to TFRecord for object_detection.
Example usage:
    python object_detection/dataset_tools/create_labelimg_tf_record.py \
        --img_dir=/home/user/data/imgs \
        --annotations_dir=/home/user/data/labels \
        --output_path=/home/user/pascal.record
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('img_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def main(_):

  img_dir = FLAGS.img_dir
  
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  annotations_dir = FLAGS.annotations_dir
  #examples_list = dataset_util.read_examples_list(examples_path)
  for idx, example in enumerate(examples_list):
    if idx % 30 == 0:
      logging.info('On image %d of %d', idx, len(examples_list))
    path = os.path.join(annotations_dir, example + '.xml')
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, FLAGS.img_dir, label_map_dict,
                                    FLAGS.ignore_difficult_instances)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
tf.app.run()
