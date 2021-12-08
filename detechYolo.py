import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import mysql.connector as mc
from playsound import playsound
import threading
import datetime

FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, is_ascii, \
    non_max_suppression, scale_coords, xyxy2xywh, set_logging, increment_path, \
    save_one_box, check_suffix
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


class Detech:
    conf_thres=0.7  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    view_img=False  # show results
    save_img=False  # save images (if the source is images and not video)
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=True  # do not save images/videos
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    project='runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference

    def __init__(self, weights='DetechModel.pt', source='0', imgsz='640', device='cpu', cameraName='cctv', classes=None, selectedClass=0, user_id=0) -> None:
        self.weights = weights # Name of the custom trained model
        self.source = source # Video source
        self.imgsz = imgsz # Image size
        self.device = device # Processing device
        self.cameraName = cameraName # Camera name
        self.classes = classes # Class names in the custom model
        self.th = threading.Thread(target=self.runInference, daemon=True) # Thread for inference
        self.isDetecting = False # Detecting status
        self.frame = None # Annotated frame
        self.webcam = True # Video source type
        self.names = None # Class names
        self.stride = None # Stride value
        self.ascii = None # ASCII class names
        self.pt = None # Pytorch model
        self.dataset = None # Processed video source
        self.bs = None # Batch Size
        self.model = None # Processed custom model
        self.save_dir = None # Directory for saving
        self.vid_path = None # Path for video file
        self.vid_writer = None # Video writer variable
        self.Notifies = False # True to play sound
        self.hasViolator = False # Existence of violator in the frame
        self.classNames = {"" : 0} # Class name dictionary
        self.checker = {"": 0} # Checker dictionary
        self.show_res = False # Show result
        self.selectedClass = selectedClass # Selected non-violator class
        self.user_id = user_id # User ID of the program user


        FILE = Path(__file__).resolve()
        sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
        save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images

        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half &= self.device.type != self.device  # half precision only supported on CUDA
        pass

    # Load the trained custom model
    def loadModel(self):
        w = self.weights[0] if isinstance(self.weights, list) else self.weights
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt','']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        self.pt, saved_model = (suffix == x for x in suffixes)  # backend booleans
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

        if self.pt:
            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
            if self.half:
                self.model.half()  # to FP16
            
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.ascii = is_ascii(self.names)  # names are ascii (use PIL for UTF-8)

        # Rename the class names
        self.model.names[0] = "with both"
        self.model.names[1] = "facemask only"
        self.model.names[2] = "faceshield only"
        self.model.names[3] = "without both"

    # Load the video source
    def loadData(self):
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            self.bs = len(self.dataset)  # batch_size
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            self.bs = 1  # batch_size
        self.vid_path, self.vid_writer = [None] * self.bs, [None] * self.bs

    # Load all the requirements and start the inference
    def startInference(self):
        self.loadModel()
        self.loadData()
        self.isDetecting = True
        self.th.start()

    # Stop the inference
    def stopInference(self):
        self.isDetecting = False
        self.dataset.stop()
        self.th.join()

    # Start the inference
    def runInference(self):
        fileName = ""
        hasFileName = False

        # Dictionary that contains detected class quantity for the previous frame
        self.checker = {
            "with both": 0,
            "facemask only": 0,
            "faceshield only": 0,
            "without both": 0
        }

        #Dictionary that contains detected class percentage for the current frame
        self.percentage = {
            "with both": 0,
            "facemask only": 0,
            "faceshield only": 0,
            "without both": 0
        }

        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        dt, seen = [0.0, 0.0, 0.0], 0

        for path, img, im0s, vid_cap in self.dataset:
            
            # Stop the loop if the inference is stopped
            if not self.isDetecting:
                break

            # Dictionary that contains detected class quantity for the current frame
            self.classNames = {
            "with both": 0,
            "facemask only": 0,
            "faceshield only": 0,
            "without both": 0
            }

            self.quantity = 0
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            if self.pt:
                self.visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred = self.model(img, augment=self.augment, visualize=self.visualize)[0]

            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                detections = 0
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), self.dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(self.dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, pil=not ascii)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print and save results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        detections += int(n) # Add current class detection quantity to the total detections quantity
                        self.classNames[self.names[int(c)]] = int(n) # Current class quantity


                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {100.0*conf:.1f}%')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()

                # Check violators and changes in qty
                for violation in self.classNames:
                    if detections == 0: # Check if detections is 0 to avoid zero division
                        self.percentage[violation] = "0.0"
                    else:
                        self.percentage[violation] = f"{100*float(self.classNames[violation]/detections):.1f}" # Get the percentage of the protection usage by dividing each detection with th summation of all detections
                    if violation != self.model.names[self.selectedClass] and self.classNames[violation] != self.checker[violation] and self.classNames[violation] != 0: # Check if the detected class is a violation, check if the quantity of the violators is the same with the previous frame, check if the quantity is 0
                        
                        # Play different sound effects for each violation
                        if violation == "facemask only":
                            playsound('sounds/face-mask-only.wav', False) # Play sound
                        elif violation == "faceshield only":
                            playsound('sounds/face-shield-only.wav', False) # Play sound
                        else:
                            playsound('sounds/no-both.wav', False) # Play sound
                        
                        #Check if there is already a file name for the screenshot
                        if hasFileName == False:
                            dateFormat = datetime.datetime.now() # Get current date and time
                            dateString = dateFormat.strftime("%d-%m-%Y-%H-%M-%S-%f") # Set format for date and time
                            fileName = f"violators/{dateString}.jpg" # File name string
                            hasFileName = True
                        self.saveScreenshot(fileName, im0) # save screenshot to local device
                        self.screenshotDb(violation, self.classNames[violation], self.cameraName, fileName) # Save violation to the database
                
                # Write percentage value
                space = 15 # Vertical space for each detection class text

                # Write detected class percentage in the frame
                for classnames in self.classNames:
                    cv2.putText(im0, f"{classnames}: {self.percentage[classnames]}%", (0,space), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    space += 20


                self.frame = im0 # Set the current frame as the value of class variable frame
                self.checker = self.classNames # Update the value of the checker variable
                    
                hasFileName = False # Reset file name for different frame

                if self.view_img and self.show_res:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                
                # print("")

    # Save screenshot to local
    def saveScreenshot(self, name, img):
        cv2.imwrite(name, img)

    # save info to DB
    def screenshotDb(self, violation, quantity, camera, name):
        mydb = mc.connect(
            host="localhost",
            user="root",
            password="",
            database="detech"
        )

        myCursor = mydb.cursor()
        insert = "INSERT INTO violators (user_id, violation, quantity, camera, filename) VALUES (%s, %s, %s, %s, %s)"
        value = (self.user_id, violation, int(f"{quantity}"), camera, name)
        myCursor.execute(insert, value)
        mydb.commit()



