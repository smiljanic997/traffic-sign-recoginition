import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

class TrafficSignDetector:

    def __init__(self):
        # The pretrained neural network model for traffic sign classification is loaded
        self.__model = load_model('./ssar_tsd_model2.h5')
        
        # HSV color extraction parameters
        self.__red_bounds = np.array([(np.array([0, 70, 200]), np.array([10, 255, 255])),(np.array([170, 70, 200]),np.array([180, 255, 255]))])
        self.__blue_bounds = np.array([(np.array([94, 127, 200]), np.array([126, 255, 255]))])
       
        # For each frame there is an array of traffic sign classes detected in that frame.
        # When the deque gets full, the oldest frame array in the deque is removed and the newest frame array is inserted (FIFO method)
        self.__num_frames = 10
        self.__frame_predictions = deque([np.array([]) for i in range(self.__num_frames)], maxlen=self.__num_frames)
        
        # The dictionary holds the number of detections for each class in the last frame_count frames
        self.__class_count = {key:0 for key in range(43)}
        
    def __extract_signs(self, rgb_frame, color_bounds):
        HEIGHT, WIDTH = rgb_frame.shape[0], rgb_frame.shape[1] 
        p = 0.003
        sp = HEIGHT*WIDTH*p
        PADDING = 5
        
        # Convert RGB frame to HSV for easy color extraction
        hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
        
        # Create a mask to separate wanted color regions from the background
        mask_n = [cv2.inRange(hsv_frame, color_bounds[i][0], color_bounds[i][1]) for i in range(len(color_bounds))]
        mask = sum(mask_i for mask_i in mask_n)

        # Color extraction
        hsv_res = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask)

        # Dilating
        gray_res = cv2.dilate(hsv_res[:,:,2], None, iterations=5)
        #gray_res = cv2.dilate(cv2.cvtColor(cv2.cvtColor(hsv_res, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), None, iterations=5)

        # Find contours in an image
        contour_info = []
        contours, _ = cv2.findContours(gray_res, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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
        
        #mask = np.zeros(gray_res.shape)
        #for bc in biggest_contours:
        #    cv2.fillConvexPoly(mask, bc[0], (255))
        #mask = cv2.dilate(mask, None, iterations=10)
        #mask = cv2.erode(mask, None, iterations=10)
        #mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        #plt.rcParams["figure.figsize"] = (15,15)
        #plt.figure()
        #_, axarr = plt.subplots(1,2)
        #axarr[0].imshow(gray_res)
        #axarr[1].imshow(mask)
        #plt.show()
        #plt.rcParams["figure.figsize"] = (6.4, 4.8)
        
        # Find the coordinates of each contour and their size
        x, y, w, h = np.zeros((NUM_CONTOURS), dtype=np.int), np.zeros((NUM_CONTOURS), dtype=np.int), np.zeros((NUM_CONTOURS), dtype=np.int), np.zeros((NUM_CONTOURS), dtype=np.int)
        for i in range(NUM_CONTOURS):
            x[i], y[i], w[i], h[i] = cv2.boundingRect(biggest_contours[i][0])
        
        # Extract the regions from original frame according to detected contours
        return [rgb_frame[y[i]-(PADDING if (y[i]-PADDING)>=0 else y[i]):y[i]+h[i]+(PADDING if (y[i]+h[i]+PADDING)<=HEIGHT else HEIGHT-y[i]-h[i]), x[i]-(PADDING if (x[i]-PADDING)>=0 else x[i]):x[i]+w[i]+(PADDING if (x[i]+w[i]+PADDING)<=WIDTH else WIDTH-x[i]-w[i]), :] for i in range(NUM_CONTOURS)]

    def __predict_class(self, rgb_region):
        img = Image.fromarray(rgb_region, 'RGB')
        img = img.resize((30, 30))
        img = np.expand_dims(img, axis=0)
        return np.argmax(self.__model.predict(np.array(list(img))), axis=1)[0] # prediction

    def predict(self, rgb_frame):
        red_regions = self.__extract_signs(rgb_frame, self.__red_bounds)
        blue_regions = self.__extract_signs(rgb_frame, self.__blue_bounds)
        
        predicted_classes = []
        
        for region in red_regions:
            predicted_class = self.__predict_class(region)
            self.__class_count[predicted_class] += 1
            predicted_classes.append(predicted_class)
        for region in blue_regions:
            predicted_class = self.__predict_class(region)
            self.__class_count[predicted_class] += 1
            predicted_classes.append(predicted_class)
        
        # Remove frame class counts from the dictionary for the frame that is getting removed from the deque
        prev_classes = self.__frame_predictions.popleft()
        for prev_class in prev_classes:
            self.__class_count[prev_class] -= 1
        
        self.__frame_predictions.append(np.asarray(predicted_classes))
        
        keys = []
        values = []
        for k,v in self.__class_count.items():
            if v >= np.round(self.__num_frames*0.7):
                keys.append(k)
                values.append(v)
        
        return [k for _, k in sorted(zip(values,keys), key=lambda pair: pair[0]) if k not in [6, 12, 32, 41, 42]][:3]

    def get_num_of_frames(self):
        return self.__num_frames
    
    def set_num_of_frames(self, num):
        self.__num_frames = num
    