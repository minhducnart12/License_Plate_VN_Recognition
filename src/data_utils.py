import numpy as np
import cv2


def get_digits_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)
    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train digits data: ', len(data_train))

    return data_train


def get_alphas_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)

    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train alphas data: ', len(data_train))

    return data_train


def get_labels(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    return [line.strip() for line in lines]


def draw_labels_and_boxes(image, lw, labels, boxes, color=(255, 0, 0), txt_color=(255, 255, 255)):
    # x_min = round(boxes[0])
    # y_min = round(boxes[1])
    # x_max = round(boxes[0] + boxes[2])
    # y_max = round(boxes[1] + boxes[3])
    # x_min = int(boxes[0])
    # y_min = int(boxes[1])
    # x_max = int(boxes[2]) 
    # y_max = int(boxes[3])

    # image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    # image = cv2.putText(image, labels, (x_min - 20, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25, color=(0, 0, 255), thickness=2)


    p1, p2 = (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    tf = max(lw - 1,1) #font thickness
    w, h = cv2.getTextSize(labels, 0, fontScale=lw/3, thickness=2)[0]  # text width, height
    outside = p1[1] - h - 3 >= 0  # label fits outside box
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image, labels, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    return image


def get_output_layers(model):
    layers_name = model.getLayerNames()
    output_layers = [layers_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    return output_layers


def order_points(coordinates):
    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = coordinates

    # top left - top right - bottom left - bottom right
    rect[0] = np.array([round(x_min), round(y_min)])
    rect[1] = np.array([round(x_min + width), round(y_min)])
    rect[2] = np.array([round(x_min), round(y_min + height)])
    rect[3] = np.array([round(x_min + width), round(y_min + height)])

    return rect


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

#Crop License Plate
# def LpCrop(image,bboxes_yolo):
#     lpcrops = []
#     for bbox_yolo in bboxes_yolo:
#         for i in range(len(bbox_yolo)):
#             x_min, y_min, x_max, y_max =bbox_yolo
#             lpcrop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
#         lpcrops.append(lpcrop)
#     return lpcrops
    