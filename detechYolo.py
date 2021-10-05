import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

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
    weights='DetechModel.pt'  # model.pt path(s)
    source='0'  # file/dir/URL/glob, 0 for webcam
    imgsz=640  # inference size (pixels)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_txt=False  # save results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    update=False  # update all models
    project='runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = True
    names = None
    stride = None
    ascii = None
    view_img = None
    pt = None
    dataset = None
    bs = None
    model = None
    save_dir = None
    vid_path = None
    vid_writer = None

    def __init__(self, weights, source, imgsz, device) -> None:
        self.weights = weights
        self.source = source
        self.imgsz = imgsz
        self.device = device

        FILE = Path(__file__).resolve()
        sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

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

    def runInference(self):
        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        dt, seen = [0.0, 0.0, 0.0], 0

        for path, img, im0s, vid_cap in self.dataset:
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

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

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

                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if self.vid_path[i] != save_path:  # new video
                            self.vid_path[i] = save_path
                            if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                self.vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        self.vid_writer[i].write(im0)

    

