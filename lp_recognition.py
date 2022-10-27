import numpy as np
import cv2
import imutils
from skimage import measure

from skimage.filters import threshold_local

from my_yolov6 import my_yolov6
from src.char_classification.model import CNN_Model
from src.data_utils import convert2Square, draw_labels_and_boxes


ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

CHAR_CLASSIFICATION_WEIGHTS = 'weights/weight.h5'
WEIGHT_PATH = 'weights/license_plate_yolov6.pt'
DATA_YAML_PATH = 'data/mydataset.yaml'

def maximizeContrast(imgGrayscale):
    #Làm cho độ tương phản lớn nhất 
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    #cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    #cv2.imwrite("blackhat.jpg",imgBlackHat)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat) 
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    #cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    #Kết quả cuối là ảnh đã tăng độ tương phản 
    return imgGrayscalePlusTopHatMinusBlackHat

def LpCrop(image,bbox_yolo):
    x_min, y_min, x_max, y_max =bbox_yolo
    lpcrop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    return lpcrop

class E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.candidates = []
        self.weight_path = WEIGHT_PATH
        self.data_yaml = DATA_YAML_PATH
        self.yolov6_model = my_yolov6(self.weight_path,"cpu",self.data_yaml , 640, False)
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)


    def extractLP(self):
        # detect license plate by yolov6
        bboxes_yolo = self.yolov6_model.infer(self.image, conf_thres=0.5, iou_thres=0.45)[:,:4]
        if len(bboxes_yolo) == 0:
            ValueError('No images detected')
        for bbox_yolo in bboxes_yolo:
            yield bbox_yolo

    def predict(self, image):
        # Input image or frame
        self.image = image

        for bbox_yolo in self.extractLP():
            self.candidates = []

            # crop number plate
            LpRegion = LpCrop(self.image, bbox_yolo)

            # segmentation
            self.segmentation(LpRegion)

            # recognize characters
            self.recognizeChar()

            # format and display license plate
            license_plate = self.format()

            # draw labels
            self.image = draw_labels_and_boxes(self.image, max(round(sum(self.image.shape) / 2 * 0.003), 2), license_plate, bbox_yolo)

        return self.image


    def segmentation(self, LpRegion):
            # apply thresh to extracted licences plate
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        V = cv2.equalizeHist(V)
        V2 = maximizeContrast(V)

            # adaptive threshold
        T = threshold_local(V2, 15, offset=10, method="gaussian")
        thresh = (V2 > T).astype("uint8") * 255

            # convert black pixel of digits to white pixel
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)
    
            # connected components analysis
        labels = measure.label(thresh, connectivity=2, background=0)

            # loop over the unique components
        for label in np.unique(labels):
                # if this is background label, ignore it
            if label == 0:
                continue

                # init mask to store the location of the character candidates
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

                # find contours from mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                    # rule to determine characters
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                keepAspectRatio = 0.1 < aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.35

                if keepAspectRatio and keepSolidity and keepHeight:
                        # extract characters
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        characters = []
        coordinates = []

        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)
        result = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)
    
        self.candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31:    # if is background or noise, ignore it
                continue
            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))
        # print(self.candidates)

    def format(self):
        first_line = []
        second_line = []

        for candidate, coordinate in self.candidates:
            if coordinate[0] < 100:
                first_line.append((candidate, coordinate))
            else:
                second_line.append((candidate, coordinate))

        first_line = sorted(first_line, key=lambda x:x[1][1])
        second_line = sorted(second_line, key=lambda x:x[1][1])

        if len(second_line) == 0:  # if license plate has 1 line
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:   # if license plate has 2 lines
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])

        return license_plate