import numpy as np
import collections
import operator
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tflite_runtime.interpreter import Interpreter, load_delegate
from utils import is_device_present


class TrafficSignDetector:

    def __init__(self):
        # Whether TensorFlow operations will run on Coral TPU
        self.__run_on_coral = is_device_present('1a6e:089a') or is_device_present('18d1:9302')

        # Pretrained neural network model for traffic sign classification is loaded
        if self.__run_on_coral == False:
            self.__model = load_model('ssar_tsr_model_no_aug.h5')
        else:
            self.__interpreter = self.__make_interpreter()
            self.__interpreter.allocate_tensors()
        
        # HSV color extraction parameters
        self.__red_bounds = np.array([(np.array([0, 128, 100]), np.array([10, 255, 255])),
                                      (np.array([170, 128, 100]), np.array([180, 255, 255]))])
        self.__blue_bounds = np.array([(np.array([95, 128, 100]), np.array([125, 255, 255]))])

        # For each frame there is an array of traffic sign classes detected in that frame.
        # When the deque gets full, the oldest frame array in the deque is
        # removed and the newest frame array is inserted (FIFO method)
        self.__num_frames = 5
        self.__frame_predictions = collections.deque([np.array([]) for i in range(self.__num_frames)], maxlen=self.__num_frames)

        # The dictionary holds the number of detections for each class in the
        # last __num_frames frames
        self.__class_count = { key: 0 for key in range(42) }
    
    def coral_detected(self):
        return self.__run_on_coral

    def __make_interpreter(self):
        """Loads quantized model and standard."""
        EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
        MODEL_PATH = 'ssar_tsr_model_no_aug_quant_edgetpu.tflite'
        model_file, *device = MODEL_PATH.split('@')
        return Interpreter(model_path = model_file,
                            experimental_delegates = [load_delegate(EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})])

    def __input_tensor(self):
        """Returns input tensor view as numpy array of shape (height, width, 3)."""
        tensor_index = self.__interpreter.get_input_details()[0]['index']
        return self.__interpreter.tensor(tensor_index)()[0]

    def __output_tensor(self):
        """Returns dequantized output tensor."""
        output_details = self.__interpreter.get_output_details()[0]
        output_data = np.squeeze(self.__interpreter.tensor(output_details['index'])())
        return output_data

    def __set_input(self, data):
        """Copies data to input tensor."""
        self.__input_tensor()[:, :] = data

    def __get_output(self, top_k=1, score_threshold=0.0):
        """Returns no more than top_k classes with score >= score_threshold."""
        Class = collections.namedtuple('Class', ['id', 'score'])
        scores = self.__output_tensor()
        classes = [
            Class(i, scores[i])
            for i in np.argpartition(scores, -top_k)[-top_k:]
            if scores[i] >= score_threshold
        ]
        return sorted(classes, key=operator.itemgetter(1), reverse=True)
        
    def __input_size(self):
        """Returns an input image size as (width, height) tuple."""
        _, height, width, _ = self.__model.get_layer(index=0).input_shape if self.__run_on_coral == False else self.__interpreter.get_input_details()[0]['shape']
        return width, height

    def __extract_regions(self, rgb_frame, color_bounds):
        """Takes a camera frame and extracts regions of interest (traffic signs)."""
        HEIGHT, WIDTH = rgb_frame.shape[0], rgb_frame.shape[1] 
        p = 0.003
        sp = HEIGHT*WIDTH*p

        # Convert RGB frame to HSV for easy color extraction
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

        # Create a mask to separate wanted color regions from the background
        mask = 0
        for i in range(len(color_bounds)):
            mask_n = [cv2.inRange(hsv_frame, color_bounds[i][n][0], color_bounds[i][n][1]) for n in range(len(color_bounds[i]))]
            mask += sum(mask_i for mask_i in mask_n)
        mask = cv2.dilate(mask, None, iterations=1)

        # Find contours in an image
        contour_info = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c)
            ))
        contour_info = [ci for ci in contour_info if ci[1] is False and ci[2]>sp]
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        NUM_CONTOURS = np.min([3,len(contour_info)])
        if NUM_CONTOURS == 0:
            return []
        biggest_contours = contour_info[0:NUM_CONTOURS]

        # Find the coordinates of each contour and their size
        x, y, w, h = np.zeros((NUM_CONTOURS), dtype=np.int), np.zeros((NUM_CONTOURS), dtype=np.int), np.zeros((NUM_CONTOURS), dtype=np.int), np.zeros((NUM_CONTOURS), dtype=np.int)
        for i in range(NUM_CONTOURS):
            x[i], y[i], w[i], h[i] = cv2.boundingRect(biggest_contours[i][0])

        # Extract the regions from original frame according to detected contours
        return [rgb_frame[y[i]:y[i]+h[i], x[i]:x[i]+w[i], :] for i in range(NUM_CONTOURS)]
        
    def __predict_class(self, rgb_region):
        """Classifies a single image on a CPU."""
        img = Image.fromarray(rgb_region, 'RGB')
        img = img.resize(self.__input_size(), Image.ANTIALIAS)
        img = np.expand_dims(img, axis=0)
        return np.argmax(self.__model.predict(np.array(list(img))), axis=1)[0]

    def __predict_class_on_TPU(self, rgb_region):
        """Classifies a single image on a Coral TPU."""
        img = Image.fromarray(rgb_region, 'RGB')
        img = img.resize(self.__input_size(), Image.ANTIALIAS)
        img = np.expand_dims(img, axis=0)
        self.__set_input(img)
        self.__interpreter.invoke()
        return self.__get_output()

    def predict(self, rgb_frame):
        """
            Parameters:
                rgb_frame - frame taken from a camera module. Must be RGB.
            Returns a list of predicted signs (up to 3).
        """
        regions_of_interest = self.__extract_regions(rgb_frame, [self.__red_bounds, self.__blue_bounds])

        predicted_classes = []

        for region in regions_of_interest:
            predicted_class = self.__predict_class_on_TPU(region)[0][0] if self.__run_on_coral == True else self.__predict_class(region)
            self.__class_count[predicted_class] += 1
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
            # 0.7 means that the detected sign has to be in at least 70%
            # of the last __num_frames frames for it to be returned
            if v >= np.round(self.__num_frames * 0.7):
                keys.append(k)
                values.append(v)

        # sort and return up to 3 signs(this could be empty list)
        return [k for _, k in sorted(zip(values, keys), key=lambda pair: pair[0])][:3]
