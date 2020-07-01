import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tensorflow.lite as tflite
import collections
import operator
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter


class TrafficSignDetector:

    def __init__(self, run_on_coral=True):
        # The pretrained neural network model for traffic sign classification is loaded
        self.__run_on_coral = run_on_coral
        if run_on_coral == False:
            self.__model = load_model('small_ssar_tsd.h5')
        else:
            self.interpreter = self.__make_interpreter()
            self.interpreter.allocate_tensors()
        # HSV color extraction parameters
        self.__red_bounds = np.array([(np.array([0, 70, 200]), np.array(
            [10, 255, 255])), (np.array([170, 70, 200]), np.array([180, 255, 255]))])
        self.__blue_bounds = np.array(
            [(np.array([94, 127, 200]), np.array([126, 255, 255]))])

        # For each frame there is an array of traffic sign classes detected in that frame.
        # When the deque gets full, the oldest frame array in the deque is
        # removed and the newest frame array is inserted (FIFO method)
        self.__num_frames = 5
        self.__frame_predictions = deque([np.array([]) for i in range(self.__num_frames)], maxlen=self.__num_frames)

        # The dictionary holds the number of detections for each class in the
        # last frame_count frames
        self.__class_count = { key: 0 for key in range(43) }

    def __make_interpreter(self):
        """Loads quantized model and standard """
        EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
        MODEL_PATH = 'small_quant_edgetpu.tflite'
        model_file, *device = MODEL_PATH.split('@')
        return Interpreter(
            model_path=model_file,
            experimental_delegates=[
                load_delegate(EDGETPU_SHARED_LIB,
                              {'device': device[0]} if device else {})
            ])

    def __input_tensor(self):
        """Returns input tensor view as numpy array of shape (height, width, 3)."""
        tensor_index = self.interpreter.get_input_details()[0]['index']
        return self.interpreter.tensor(tensor_index)()[0]

    def __output_tensor(self):
        """Returns dequantized output tensor."""
        output_details = self.interpreter.get_output_details()[0]
        output_data = np.squeeze(
            self.interpreter.tensor(
                output_details['index'])())
        return output_data

    def __set_input(self, data):
        """Copies data to input tensor."""
        self.__input_tensor()[:, :] = data

    def __input_size(self):
        """Returns input image size as (width, height) tuple."""
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
        return width, height

    def __get_output(self, top_k=1, score_threshold=0.0):
        """Returns no more than top_k classes with score >= score_threshold."""
        Class = collections.namedtuple('Class', ['id', 'score'])
        scores = self.__output_tensor()
        # print(scores)
        classes = [
            Class(i, scores[i])
            for i in np.argpartition(scores, -top_k)[-top_k:]
            if scores[i] >= score_threshold
        ]
        return sorted(classes, key=operator.itemgetter(1), reverse=True)

    def __extract_signs(self, rgb_frame, color_bounds):
        """Takes the whole frame from camera and extracts region of interest."""
        HEIGHT, WIDTH = rgb_frame.shape[0], rgb_frame.shape[1]
        p = 0.003
        sp = HEIGHT * WIDTH * p
        PADDING = 5

        # Convert RGB frame to HSV for easy color extraction
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

        # Create a mask to separate wanted color regions from the background
        mask_n = [
            cv2.inRange(
                hsv_frame,
                color_bounds[i][0],
                color_bounds[i][1]) for i in range(
                len(color_bounds))]
        mask = sum(mask_i for mask_i in mask_n)

        # Color extraction
        hsv_res = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

        # Dilating
        gray_res = cv2.dilate(hsv_res[:, :, 2], None, iterations=5)

        # Find contours in an image
        contour_info = []
        contours, _ = cv2.findContours(
            gray_res, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c)
            ))
        contour_info = [
            ci for ci in contour_info if ci[1] is False and ci[2] > sp]
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        NUM_CONTOURS = np.min([3, len(contour_info)])
        if NUM_CONTOURS == 0:
            return []
        biggest_contours = contour_info[0:NUM_CONTOURS]

        # Find the coordinates of each contour and their size
        x, y, w, h = np.zeros(
            (NUM_CONTOURS), dtype=np.int), np.zeros(
            (NUM_CONTOURS), dtype=np.int), np.zeros(
            (NUM_CONTOURS), dtype=np.int), np.zeros(
                (NUM_CONTOURS), dtype=np.int)
        for i in range(NUM_CONTOURS):
            x[i], y[i], w[i], h[i] = cv2.boundingRect(biggest_contours[i][0])

        # Extract the regions from original frame according to detected
        # contours
        return [rgb_frame[y[i] - (PADDING if (y[i] - PADDING) >= 0 else y[i]):
                          y[i] + h[i] + (PADDING if (y[i] + h[i] + PADDING) <= HEIGHT else HEIGHT - y[i] -h[i]), x[i] -
                          (PADDING if (x[i] -PADDING) >= 0 else x[i]):x[i] + w[i] + 
                          (PADDING if (x[i] + w[i] + PADDING) <= WIDTH else WIDTH - x[i] - w[i]), :] for i in range(NUM_CONTOURS)]

    def __predict_class(self, rgb_region):
        """Classifies single image on CPU."""
        img = Image.fromarray(rgb_region, 'RGB')
        img = img.resize((30, 30))
        img = np.expand_dims(img, axis=0)
        return np.argmax(self.__model.predict(np.array(list(img))), axis=1)[0]  # prediction

    def __predict_class_on_TPU(self, rgb_region):
        """Classifies single image on Coral TPU"""
        img = Image.fromarray(rgb_region, 'RGB')
        size = self.__input_size()
        img = img.resize(size, Image.ANTIALIAS)
        img = np.expand_dims(img, axis=0)
        self.__set_input(img)
        self.interpreter.invoke()
        return self.__get_output()

    def predict(self, rgb_frame):
        """
           Parameters:
            rgb_frame - frame taken from camera module. Must be RGB.
           Returns list of predicted signs. List can be empty and max length is 3.
        """
        red_regions = self.__extract_signs(rgb_frame, self.__red_bounds)
        blue_regions = self.__extract_signs(rgb_frame, self.__blue_bounds)

        predicted_classes = []

        # flow_from_directory messes up the labels, so this dict is used to get the actual prediction
        dictt = {
            0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18,
            11: 19, 12: 2, 13: 20, 14: 21, 15: 22, 16: 23, 17: 24, 18: 25, 19: 26, 20: 27,
            21: 28, 22: 29, 23: 3, 24: 30, 25: 31, 26: 32, 27: 33, 28: 34, 29: 35, 30: 36,
            31: 37, 32: 38, 33: 39, 34: 4, 35: 40, 36: 41, 37: 5, 38: 6, 39: 7, 40: 8, 41: 9
            }

        for region in red_regions:
            # predicted_class = self.__predict_class(region)
            predicted_class = dictt[int(self.__predict_class_on_TPU(region)[0][0])]
            self.__class_count[int(predicted_class)] += 1
            predicted_classes.append(predicted_class)
        for region in blue_regions:
            # predicted_class = self.__predict_class(region)
            predicted_class = dictt[int(self.__predict_class_on_TPU(region)[0][0])]
            self.__class_count[int(predicted_class)] += 1
            predicted_classes.append(predicted_class)

        # Remove frame class counts from the dictionary for the frame that is
        # getting removed from the deque
        prev_classes = self.__frame_predictions.popleft()
        for prev_class in prev_classes:
            self.__class_count[prev_class] -= 1

        self.__frame_predictions.append(np.asarray(predicted_classes))

        keys = []
        values = []
        for k, v in self.__class_count.items():
            if v >= np.round(self.__num_frames * 0.7):
                keys.append(k)
                values.append(v)

        return [k for _, k in sorted(zip(values, keys), key=lambda pair: pair[0])][:3]

    def get_num_of_frames(self):
        return self.__num_frames

    def set_num_of_frames(self, num):
        self.__num_frames = num
