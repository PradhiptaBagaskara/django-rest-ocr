from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import six

import cv2
import numpy as np
import tensorflow as tf
from api.utils import label_map_util
from PIL import Image
import pytesseract

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    if image.getdata().mode != "RGB":
        image = image.convert('RGB')
    np_array = np.array(image.getdata())
    reshaped = np_array.reshape((im_height, im_width, 3))
    return reshaped.astype(np.uint8)


def detect_and_create_boxes(image, model_dir):
    """
        method force detecting image
    Args:
        image: PIL Image
        model_dir: checkpoint directory

    Return:
    Dict detected image data.

    """

    label_path = os.path.join(model_dir, 'label_map.pbtxt')
    ckpt_path = os.path.join(model_dir, 'frozen_inference_graph.pb')

    output = {}

    tf.reset_default_graph()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            serialize_graph = fid.read()
            od_graph_def.ParseFromString(serialize_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(label_path)
    label_map_dict = label_map_util.get_label_map_dict(label_path)
    num_classes = len(label_map_dict)

    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        tf_session = tf.InteractiveSession()  # faster session
        ops = tf.get_default_graph().get_operations()
        tensor_dict = {}
        image_np = load_image_into_numpy_array(image)
        image_for_saved = image_np
        tensor_dict = {
            'detection_boxes': detection_graph.get_tensor_by_name('detection_boxes:0'),
            'detection_scores': detection_graph.get_tensor_by_name('detection_scores:0'),
            'detection_classes': detection_graph.get_tensor_by_name('detection_classes:0'),
            'num_detections': detection_graph.get_tensor_by_name('num_detections:0')
        }

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        output_dict = tf_session.run(tensor_dict,
                                        feed_dict={image_tensor: image_np_expanded})

        num_detections = int(output_dict['num_detections'][0])
        classes = output_dict['detection_classes'][0].astype(np.uint8)
        boxes = output_dict['detection_boxes'][0]
        scores = output_dict['detection_scores'][0]

        result = {
            'image': image_np,
            'boxes': boxes,
            'classes': classes,
            'scores': scores,
            'category_index': category_index,
            'num_detections': num_detections
        }

    tf_session.close()
    return result

def image_smoothening(img, BINARY_THREHOLD = 180):
    BINARY_THREHOLD = 180
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def img_gray(img_array):
    """
    Args:
      numpy_array
    """ 
    img = Image.fromarray(img_array)   
    gray = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    thres, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    img = image_smoothening(filtered)

    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.bitwise_or(img, closing)
    return gray, thresh

def crop_image(image,
                ymin,
                xmin,
                ymax,
                xmax,
                label_name,
                use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  Return:
    tuple ('label_name', (image_array, data)).    
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')

  arr = np.array(image_pil) # convert image to array for croping
  im_width, im_height = image_pil.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    a,b,c,d = int(left), int(right), int(top), int(bottom)
    arr = arr[c:d,a:b]
    data = {
        'ymin': top,
        'xmin': left,
        'ymax': bottom,
        'xmax': right,
    }
    return (label_name, (arr, data))

def list_cropped_tuple(
    image,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    min_score_thresh=.5,

    ):
    """Tupple of label names and img np array.

    Args:
        image: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
        use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
        max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
        min_score_thresh: minimum score threshold for a box to be visualized      
    

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_label_str_map = collections.defaultdict(list)
    box_to_instance_masks_map = {}
    listOfcroped = []

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            label_name = "N/A"
            if classes[i] in six.viewkeys(category_index):
                label_name = category_index[classes[i]]['name']
            # print(label_name)
            box_to_label_str_map[box].append(label_name)


    # looping label
    for box, label in box_to_label_str_map.items():
            ymin, xmin, ymax, xmax = box
           
            croped = crop_image(
            image,
                ymin,
                xmin,
                ymax,
                xmax,
                label_name=label[0],
                use_normalized_coordinates=use_normalized_coordinates)

            listOfcroped.append(croped)

    return listOfcroped


def ocr_label_to_dict(image, 
              model_dir='/model/ktp', 
              label_path='ktp.pbtxt', 
              tess_config=''
               ):
    """
    Args: 

      @image : Image.open(image) #PIL image
      @ckpt_path : model path
      @label_path : label path
      @img_name : boxed image name
      @save_bounding_img : save boxed image default False
      @use_tf_version : tensorflow version. default 1,
      @tess_config : tesseract config

    Return:
    
       dict data

    """
    w, h = image.size
    detect = detect_and_create_boxes(image,model_dir)
    croped = list_cropped_tuple(
        image=detect['image'],
        boxes=detect['boxes'],
        classes=detect['classes'],
        scores=detect['scores'],
        category_index=detect['category_index']

    )
    result = {
        'width': w,
        'height': h,
        'data': None
    }
    tmp = {}
    label = label_map_util.get_label_map_dict(os.path.join(model_dir, 'label_map.pbtxt'))

    for i in label:
        tmp[i] = []
    for label,(img,data) in croped:
        gray, thresh = img_gray(img)
        text = str(pytesseract.image_to_string((thresh), config=tess_config))
        text = dict(text=text)
        data.update(text)
        tmp[label].append(data)
    result['data'] = tmp
    return result




