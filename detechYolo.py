import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import mysql.connector as mc
from playsound import playsound
import threading

FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, colorstr, is_ascii, \
    non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box, check_suffix
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


class Detech:
    conf_thres=0.7  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_img=False
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
        self.weights = weights
        self.source = source
        self.imgsz = imgsz
        self.device = device
        self.cameraName = cameraName
        self.classes = classes
        self.th = threading.Thread(target=self.runInference, daemon=True)
        self.isDetecting = False
        self.frame = None
        self.webcam = True
        self.names = None
        self.stride = None
        self.ascii = None
        self.pt = None
        self.dataset = None
        self.bs = None
        self.model = None
        self.save_dir = None
        self.vid_path = None
        self.vid_writer = None
        self.Notifies = False
        self.hasViolator = False
        self.classNames = {"" : 0}
        self.checker = {"": 0}
        self.show_res = False
        self.selectedClass = selectedClass
        self.user_id = user_id


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
        self.model.names[0] = "with both"
        self.model.names[1] = "facemask only"
        self.model.names[2] = "faceshield only"
        self.model.names[3] = "without both"
        print("Class names: ",self.model.names)

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

    def startInference(self):
        self.loadModel()
        self.loadData()
        self.isDetecting = True
        self.th.start()
        # self.runInference()

    def stopInference(self):
        self.isDetecting = False
        self.th.join()

    def runInference(self):
        fileName = ""
        hasFileName = False

        self.checker = {
            "with both": 0,
            "facemask only": 0,
            "faceshield only": 0,
            "without both": 0
        }

        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        dt, seen = [0.0, 0.0, 0.0], 0

        for path, img, im0s, vid_cap in self.dataset:
            
            if not self.isDetecting:
                print("breaking")
                break

            self.classNames = {
            "with both": 0,
            "facemask only": 0,
            "faceshield only": 0,
            "without both": 0
            }

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

                        # If violator detected
                        if self.names[int(c)] != self.model.names[self.selectedClass]:
                            self.classNames[self.names[int(c)]] = int(n)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference-only)
                print(f'{s}Done. ({t3 - t2:.3f}s)')
                print(f'{1/(t3 - t2)} FPS')

                # Stream results
                im0 = annotator.result()
                self.frame = im0

                # Check violators and changes in qty
                for violation in self.classNames:
                    if violation != self.model.names[self.selectedClass] and self.classNames[violation] != self.checker[violation] and self.classNames[violation] != 0:
                        print("New Detection")
                        playsound('sounds/notification.wav', False) # Play sound
                        if hasFileName == False:
                            fileName = "violators\\" +str(time_sync()) +".jpg"
                            hasFileName = True
                        self.saveScreenshot(fileName, im0)
                        self.screenshotDb(violation, self.classNames[violation], self.cameraName, fileName)
                
                print(self.checker)
                self.checker = self.classNames
                    
                hasFileName = False

                if self.view_img and self.show_res:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                
                print("")

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
        insert = "INSERT INTO violators (user_id ,violation, quantity, camera, filename) VALUES (%s, %s, %s, %s, %s)"
        value = (self.user_id, violation, int(f"{quantity}"), camera, name)
        myCursor.execute(insert, value)
        mydb.commit()



