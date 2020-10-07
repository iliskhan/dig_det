import os
import cv2
import random
import imutils
import colorsys
import threading

import numpy as np
import tensorflow as tf

from PIL import Image
from time import time
from threading import Thread
from yolov3.yolov4 import Create_Yolo


VIDEOS_DIR = r"./videos"

CLASSES = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
           5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


def main():

    NUM_CLASS = CLASSES

    yolo = Create_Yolo(input_size=416, CLASSES=CLASSES)
    yolo.load_weights(f"yolov3_custom_Tiny")

    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        current_digit = None
        video_thread = None
        video_names = os.listdir(VIDEOS_DIR)
        while True:
            
            _, frame = cap.read()
            image_h, image_w, _ = frame.shape
            # lt = time()
            bboxes = detect_digit(yolo, frame, CLASSES)
            
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1:
                bbox_thick = 1
            # fontScale = 0.75 * bbox_thick

            digits = []
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                if score > 0.8:
                    class_ind = int(bbox[5])
                    digit = NUM_CLASS[class_ind]
                    
                    if current_digit != digit:
                        
                        digits.append(digit)

                        frame = cap.read()[1]

                        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), bbox_thick*2)

                        for video_name in video_names:
                            if digit == video_name.split('.')[0]:

                                if video_thread and video_thread.is_alive():
                                    video_thread.do_run = False
                                    video_thread.join()
                                    current_digit = None
                                else:
                                    video_thread = Thread(target=show_video, args=(video_name, digit, frame, coor))
                                    video_thread.start()
                                    current_digit = digit

                                break
                    if video_thread and not video_thread.is_alive():
                        current_digit = None
            k = cv2.waitKey(33)
            
            if k == 27:
                break                        
  
            cv2.imshow('frame', frame)
            

    else:
        print("Камера не найдена")

    cap.release()
    cv2.destroyAllWindows()

def show_video(video_name, digit, frame, coor):
    video_cap = cv2.VideoCapture(os.path.join(VIDEOS_DIR, video_name))
    window_name = str(digit)
    toggle = False
    (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
    t = threading.currentThread()
    while video_cap.isOpened() and getattr(t, "do_run", True):

        try:
            video_frame = video_cap.read()[1]

            video_h, video_w, _ = video_frame.shape

            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                    
            k = cv2.waitKey(33)
            
            if k == ord('c'):

                toggle = not toggle

            if toggle:
                
                video_frame[video_h - (y2 - y1):, video_w - (x2 - x1):] = frame[y1: y2, x1: x2]

            cv2.imshow(window_name, video_frame)
            if k == 27:
                
                video_cap.release()
                cv2.destroyWindow(window_name)
                break
        except:
            break
    
    video_cap.release()
    cv2.destroyWindow(window_name)

    
def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * \
        (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * \
        (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or(
        (pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(
        pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and(
        (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h,  w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def detect_digit(Yolo, frame, CLASSES, input_size=416, show=False, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):

    original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_frame), [
                                  input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # t1 = time.time()

    pred_bbox = Yolo.predict(image_data)

    # t2 = time.time()

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(
        pred_bbox, original_frame, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')


    return bboxes


if __name__ == "__main__":
    main()
