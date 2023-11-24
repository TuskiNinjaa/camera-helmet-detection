from mmdet.apis import init_detector, inference_detector
import cv2
import time
import threading
import sys

class Model():
    def __init__(self, config_file, checkpoint_file, device='cuda:0', threshold=0.5, thickness=3):
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.threshold = threshold
        self.thickness = thickness

        self.bboxes = None
        self.labels = None
        self.scores = None
    
    def draw_inference(self, frame):
        if self.bboxes is None:
            return frame
            
        for i, bbox in enumerate(self.bboxes):
            if self.scores[i] > self.threshold:
                color = (255,0,0) if self.labels[i] == 1 else (0,0,255)
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, self.thickness)
        
        return frame
    
    def inference(self, frame):
        result = inference_detector(self.model, frame)

        self.bboxes = result.pred_instances.bboxes.cpu().data.numpy().astype(int)
        self.labels = result.pred_instances.labels.cpu().data.numpy()
        self.scores = result.pred_instances.scores.cpu().data.numpy()
    

class PredictiveCamera():
    def __init__(self, camera_index, config_file, checkpoint_file, device='cuda:0', threshold=0.5, thickness=3):
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.frame = None
        self.status = None
        self.new_frame = False
        self.delay = 10**(-10)
        self.model = Model(config_file, checkpoint_file, device, threshold, thickness)

        thread_frame = threading.Thread(target=self.update, args=())
        thread_frame.daemon = True
        thread_frame.start()
        thread_model = threading.Thread(target=self.predict, args=())
        thread_model.daemon = True
        thread_model.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            self.new_frame = False
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
                self.new_frame = True
            time.sleep(self.delay)
    
    def predict(self):
        while True:
            if self.new_frame:
                self.model.inference(frame=self.frame)

    def getFrame(self):
        return self.status, self.model.draw_inference(self.frame)
        
    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()

DELAY_FPS = 0.5

def main():
    config_file = './models/rtmdet_tiny_8xb32-300e_coco_helmet.py'
    checkpoint_file = './models/rtmdet_tiny.pth'
    
    if sys.argv[1] == "cuda":
        device = "cuda"
    else:
        device = "cpu" 

    camera_index = 0
    frame_counter = 0
    fps = 0
    start_time = time.time()
    camera = PredictiveCamera(camera_index, config_file, checkpoint_file, device=device)

    while True:
        check, frame = camera.getFrame()
        if check != None:
            cv2.putText(frame, "FPS: "+str(int(fps)), (5, 25), cv2.FONT_HERSHEY_SIMPLEX , 1, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press ESC to exit", (5, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (100, 255, 0), 1, cv2.LINE_AA)
            # model.inference(frame)
            # frame = model.draw_inference(frame)
            cv2.imshow("Video", frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        frame_counter += 1
        if (time.time() - start_time) > DELAY_FPS:
            fps = frame_counter / (time.time() - start_time)
            frame_counter = 0
            start_time = time.time()

    camera.release()
    return 0

if __name__ == '__main__':
    sys.exit(main())