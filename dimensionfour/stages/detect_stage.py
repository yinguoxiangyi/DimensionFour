from imageai.Detection import ObjectDetection
import os
import sys
import cv2
import json
import numpy as np
import subprocess
import logging

from dimensionfour.stages.base_stage import BaseStage
USE_NANO = False
if USE_NANO:
    from darknet_nano import * 
else:
    from darknet_single_img import *
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class DetectStage(BaseStage):
   def __init__(self, args):
      super().__init__(args)

      if not os.path.isfile(args.input):
         print("[DetectStage] Input file %s not found" % args.input)
         sys.exit(1)
      ''' 
      yoloPath = "./run_artifacts/models/yolo.h5"
      
      if not os.path.isfile(yoloPath):
         print("[DetectStage] %s does not exist. Downloading now." % yoloPath)
         os.makedirs(os.path.dirname(yoloPath), exist_ok=True)
         subprocess.call(["wget","-q","--show-progress","https://www.dropbox.com/s/q5bsuy7ltxoxbkr/yolo.h5?dl=1","-O",yoloPath])
      '''
      self.detections = []
      self.frameCounter = 0
      self.filter =    args.filter
      self.cap = cv2.VideoCapture(args.input)
      '''
      self.detector = ObjectDetection()
      self.detector.setModelTypeAsYOLOv3()
      self.detector.setModelPath(yoloPath)
      self.detector.loadModel()
      '''
      configPath = "./cfg/yolov3.cfg"
      weightPath = "yolov3.weights"
      metaPath= "./cfg/coco.data"
      gpu_id = 0

      logging.info('Init model configPath %s, weightPath  %s,metaPath %s,'%(configPath, weightPath, metaPath))
      self.detector = init_model(configPath = configPath, weightPath = weightPath,  metaPath= metaPath, gpu_id= gpu_id,  )
      logging.info('Init done')

   def execute(self):

      frames = []
      while True:

         # get frame from the video
         hasFrame, frame = self.cap.read()

         # Stop the stage if reached end of video
         if not hasFrame:
            print("[DetectStage] Done processing %d frame(s)." % (self.frameCounter - 1))
            self.cap.release()
            break

         # Detect on every x frames and save results
         if self.frameCounter % 5 == 0:
            print("[DetectStage] Frame %d: Detecting" % self.frameCounter)
            # _, output_array = self.detector.detectObjectsFromImage(input_type="array", output_type="array", input_image=frame, minimum_percentage_probability=30)
            det_res = self.detector(frame)
            # print('det_res',det_res)
            output_array = list()
            for det in det_res:
               
               left, top, right, bottom , score , label = tuple(det)
               left, top, right, bottom , score , name = int(left), int(top), int(right), int(bottom) , int(score*100) , str(int(label))
               if name not in self.filter:
                  continue               
               each_object_details = dict()
               each_object_details["name"] = name
               each_object_details["percentage_probability"] = score
               each_object_details["box_points"] = [left, top, right, bottom ]
               output_array.append(each_object_details)
            print("object nums is %s" % (len(output_array)))
            self.detections.append(output_array)
            frames.append(frame)
         else:
            print("[DetectStage] Frame %d: Skipping" % self.frameCounter)

         self.frameCounter += 1
      
      
      self.writeArtifact(self.detections, "DetectStage.out.json", cls=NpEncoder)

      print("[DetectStage] Calculating median frame")
      medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)   
      cv2.imwrite(self.getArtifactPath("background_model.jpg"), medianFrame)
      print("[DetectStage] Finished writing median frame ,total detect frames is %d"%(len(frames)))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)