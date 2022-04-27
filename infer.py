import cv2
import time
import torch
import numpy as np
from facebox.faceboxes import FaceBoxes, load_model
from facebox.prior_box import PriorBox, cfg, decode
from facebox.py_cpu_nms import py_cpu_nms as nms


class facebox_pytorch():
    def __init__(self, device='cuda', confidence_threshold=0.05, top_k=5000, nms_threshold=0.3, keep_top_k=750,
                 vis_thres=0.5):
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)
        self.net = load_model(self.net, 'facebox/FaceBoxesProd.pth', device)
        self.net.eval()
        self.device = device
        self.resize = 1
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres

    def detect(self, srcimg):
        img = np.float32(srcimg)
        if self.resize != 1:
            img = cv2.resize(img, None, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(self.device)
        img -= (104, 117, 123)
        with torch.no_grad():
            img = torch.from_numpy(img).permute((2, 0, 1)).unsqueeze(0)
            img = img.to(self.device)
            loc, conf = self.net(img)
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / self.resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            # ignore low scores
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:self.top_k]
            boxes = boxes[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            # keep = py_cpu_nms(dets, args.nms_threshold)
            keep = nms(dets, self.nms_threshold)
            dets = dets[keep, :]

            # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :]

        drawimg, face_rois = srcimg.copy(), []
        for b in dets:
            if b[4] < self.vis_thres:
                continue
            b = list(map(int, b[:4]))
            cv2.rectangle(drawimg, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            face_rois.append(srcimg[b[1]:b[3], b[0]:b[2]])
        return drawimg, face_rois


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    facebox_detect = facebox_pytorch(device=device)
    facebox_detect.resize = 0.5
    imgpath = '/Users/cwpeng/Desktop/image/crop_photo.jpg'

    srcimg = cv2.imread(imgpath)

    start_time = time.time()
    for i in range(1000):
        drawimg, face_rois = facebox_detect.detect(srcimg)
    print(time.time() - start_time)

    cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
    cv2.imshow('detect', drawimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
