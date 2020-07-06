import io
import logging
import socketserver
from http import server
import threading
import time
import codecs
from copy import deepcopy
from video_get import VideoGet
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s {} : %(message)s'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

logging.info('Importing detection module (takes a few seconds)')
import ssar_tsd

import cv2
import numpy as np

predictions = []

lock = threading.RLock()
current_frame = None

logging.info('Setting up RPi camera')
vg = VideoGet(resolution=(640, 480), framerate=24)
vg.start()
time.sleep(2)

logging.info('Creating detector object')
detector = ssar_tsd.TrafficSignDetector()
logging.info('Coral detected - running on Coral' if detector.coral_detected()==True else 'Coral not detected - running on CPU')

def reading():
    """Function that reads frames from camera module."""
    global current_frame
    while True:
        with lock:
            current_frame = vg.read()


logging.info('Starting camera reading')
reading_thread = threading.Thread(target=reading, args=())
reading_thread.daemon = True
reading_thread.start()

def get_predictions(detector):
    """Gets predictions from predictor module. 
       Lock is used to guarantee that the frame
       won't change during deepcopy function call.
    """
    global predictions
    while True:
        with lock:
            cur_frame = deepcopy(current_frame)
        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
        predictions = detector.predict(cur_frame)
        print(predictions)
    
class StreamingHandler(server.BaseHTTPRequestHandler):
    
    def do_GET(self):
        global current_frame
        global predictions
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            f = codecs.open("page.html", 'r')
            p = f.read()
            content = p.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    frame = cv2.imencode('.jpeg', current_frame)[1].tostring()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/first_pred.png':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    if len(predictions) > 0:
                        with open('Meta/{}.png'.format(predictions[0]), 'rb') as img:
                            frame = img.read()
                    else:
                        with open('white.png', 'rb') as img:
                            frame = img.read()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/png')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/second_pred.png':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    if len(predictions) > 1:
                        with open('Meta/{}.png'.format(predictions[1]), 'rb') as img:
                            frame = img.read()
                    else:
                        with open('white.png', 'rb') as img:
                            frame = img.read()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/png')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/third_pred.png':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    if len(predictions) > 2:
                        with open('Meta/{}.png'.format(predictions[2]), 'rb') as img:
                            frame = img.read()
                    else:
                        with open('white.png', 'rb') as img:
                            frame = img.read()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/png')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


logging.info('Starting processing thread')
ext_frame_thread = threading.Thread(target=get_predictions, args=(detector,))
ext_frame_thread.daemon = True
ext_frame_thread.start()

logging.info('Setting up server')
try:
    address = ('', 2105)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
finally:
    vg.stop()

ext_frame_thread.join()
