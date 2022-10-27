
import numpy as np
import cv2
import imutils
from numpy import reshape
from skimage import measure
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

from my_yolov6 import my_yolov6


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


# #Crop License Plate
def LpCrop(image,bboxes_yolo):
    lpcrops = []
    for bbox_yolo in bboxes_yolo:
        for i in range(len(bbox_yolo)):
            x_min, y_min, x_max, y_max =bbox_yolo
            lpcrop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        lpcrops.append(lpcrop)
    return lpcrops

def imshow_components(labels):
    
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    labeled_img[label_hue==0] = 0
    return labeled_img

def convert2Square(image):
    """
    Resize non square image(height != width to square one (height == width)
    :param image: input images
    :return: numpy array
    """

    img_h = image.shape[0]
    img_w = image.shape[1]

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = np.zeros(shape=(img_h, (diff//2) + 1))

        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1

        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image

def segmentation(LpRegions):
    candidates = []
    threshs = []
    for LpRegion in LpRegions:
        # apply thresh to extracted licences plate
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        # V = imutils.resize(V, width=400)
        cv2.imshow("M",V)


        V = cv2.equalizeHist(V)
        V2 = maximizeContrast(V)
        # V2 = imutils.resize(V2, width=400)
        cv2.imshow("MM",V2)

        # adaptive threshold
        T = threshold_local(V2, 15, offset=10, method="gaussian")
        thresh = (V2 > T).astype("uint8") * 255
     
        # convert black pixel of digits to white pixel
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)
        cv2.imshow('Thresh',thresh)


        # connected components analysis
        labels = measure.label(thresh, connectivity=2, background=0)
        a = imshow_components(labels)
        cv2.imshow("Label",a)


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
                # print((x, y, w, h))
                # rule to determine characters
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                #moi
                # keepAspectRatio = aspectRatio < 1.0
                # keepSolidity = solidity > 0.15
                # keepHeight = heightRatio > 0.4 and heightRatio < 0.95

                #cu
                keepAspectRatio = 0.1 < aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.35
            
                if keepAspectRatio and keepSolidity and keepHeight:
                    # extract characters
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    candidates.append((square_candidate))

    return candidates

CHAR_CLASSIFICATION_WEIGHTS = 'weights/weight.h5'
WEIGHT_PATH = 'weights/license_plate_yolov6.pt'
DATA_YAML_PATH = 'data/mydataset.yaml'
IMAGE_PATH = './img_test/test12.jpg'


yolov6_model = my_yolov6(WEIGHT_PATH,"cpu",DATA_YAML_PATH, 640, False)

img = cv2.imread(IMAGE_PATH)

det = yolov6_model.infer(img, conf_thres=0.55, iou_thres=0.45)[:,:4]
print(len(det))
# det = yolov6_model.infer(img, conf_thres=0.55, iou_thres=0.45)[0][:4]
lpcrops = LpCrop(img,det)
# lpcrops = np.array(lpcrops)
# lpcrops = np.reshape(lpcrops,(25,96,3))
# print(lpcrops)
# lpcrops = maximizeContrast(lpcrops)
# cv2.imshow("Crop",lpcrops)
lp_seg = segmentation(lpcrops)

# show cac ky tu cua bien so xe
def show_images(img):
    f=plt.figure()
    for i in range(len(img)):
        f.add_subplot(4,len(img),i+1)
        plt.imshow(img[i])
    plt.show(block=True)
show_images(lp_seg)
cv2.waitKey(0)    
cv2.destroyAllWindows()



