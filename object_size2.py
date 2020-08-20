from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# load the image, convert it to grayscale, and blur it slightly
img_path = './images/IMG_3701_mask_part1.png'
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)  # 外部轮廓，仅保留拐点信息
cnts = imutils.grab_contours(cnts)  # grab contour

# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
# cnts[0]: num of object
(cnts, _) = contours.sort_contours(cnts)  # 排序
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    # compute the rotated bounding box of the contour

    orig = image.copy()
    # cv2.drawContours(orig, [c], 0, (128, 0, 0), 3)  # 每个c是对象完整的轮廓

    box = cv2.minAreaRect(c)  # 矩形的中心点坐标、高度宽度及倾斜角度
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)  # 矩形的四个顶点坐标
    box = np.array(box, dtype="int")

    # 画圆
    # (x, y), radius = cv2.minEnclosingCircle(c)
    # center = (int(x), int(y))
    # radius = int(radius)
    # cv2.circle(orig, center, radius, (255, 0, 0), 2)

    # 画椭圆
    ellipse = cv2.fitEllipse(c)  # 椭圆的坐标点，宽、长，角度
    cv2.ellipse(orig, ellipse, (255, 255, 0), 2)
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    # 本来从左下开始顺时针计数，现在统一为左上开始的顺时针
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    # 画拐点
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # compute the center-of-mass of the object
    (cX, cY) = midpoint((tltrX, tltrY), (blbrX, blbrY))
    # print(f'the centre of mass point is [{cX, cY}]')

    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}pix".format(dA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}pix".format(dB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
