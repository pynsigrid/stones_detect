from PIL import Image
import cv2
import numpy as np
import os
# from colorful import colorful


def auto_label(dirPic, width, height):
    file_list = os.listdir(dirPic + 'gray/')
    color = [255, 255, 255]
    print('start labelling!!!')
    for filename in file_list:
        path = ''
        path = dirPic + 'gray/' + filename
        path2 = dirPic + 'gray_256/' + filename
        try:
            image = cv2.imread(path)
            print(f'{filename} is on precessing.')

            # resize image
            width = 512
            height = 512
            dim = (width, height)
            gray = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # 单通道转换3通道
            # # gray_BGR = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # image = np.expand_dims(gray, axis=2)
            # gray_BGR = np.concatenate((image, image, image), axis=-1)
            # print('BGR is ok!')

            # OTSU阈值分割
            ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            print(f'ret = {ret}')

            # 开/闭运算
            kernel = np.ones((5, 5), np.uint8)
            kernel2 = np.ones((5, 5), np.uint8)
            opening2 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel2)

            # 色彩转换
            # col_inv = col_gray(color=color, img=image)
            cv2.imwrite(path2, gray)

        except:
            print('find .DS_Store!')
            continue


def col_gray(color, img):
    print(f'img is inversing to gray: {img.shape}')
    img_ = np.zeros([img.shape[0], img.shape[1]])
    # 3通道
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == color[0] and img[i, j, 1] == color[1] and img[i, j, 2] == color[2]:
                img_[i, j] = 128  # 二分类: stones=1, bg=0
            else:
                img_[i, j] = 0

    # 1通道
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if img[i, j] == color:
    #             img_[i, j] = 1  # 二分类: stones=1, bg=0
    #         else:
    #             img_[i, j] = 0
        # print(i, ' is ok')
    return img_


if __name__ == '__main__':
    dirPic = '/Users/panyining/Desktop/stones/dataset/0806_png/'
    # img = '/Users/panyining/Desktop/stones/dataset/0805_OTSU_REVERSE/IMG_3904.png'
    # img2 = col_gray(color=[0, 0, 0], img=cv2.imread(img))
    # auto_label(dirPic, 512, 512)
