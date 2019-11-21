# https://towardsdatascience.com/creating-ai-for-gameboy-part-2-collecting-data-from-the-screen-ccd7381a1a33
# https://towardsdatascience.com/creating-ai-for-gameboy-part-3-automating-awful-gameplay-b60fe7504e4e
# https://github.com/aaronfrederick/Fire-Emblem-AI
import time
import timeit
import cv2
import numpy as np
import pytesseract as pytesseract
import matplotlib.pyplot as plt
import PIL.ImageGrab
import PIL.ImageOps

#

def findNumbers(img_src, template_src):
    img_bw = cv2.imread(img_src)
    img_gray = cv2.cvtColor(img_bw, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_src, 0)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    pts = []
    tol = 4
    for pt in loc[1]:
        #print(template_src, pt)
        ok = True
        for pt2 in pts:
            if (pt < pt2 + tol and pt > pt2 - tol):
                ok = False
                #print(template_src, "same: ", pt, pt2, ok)
        #print(ok)
        if ok:
            pts.append(pt)
            #print(template_src, "Added:", pt)

    return pts


class Point:
    def __init__(self, number, spot):
        self.spot = spot
        self.number = number

    def __repr__(self):
        return str(self.number) + "/" + str(self.spot)


def screencap(emulatorBox = [20, 11, 640, 580], scoreBox = [460, 122, 585, 145]):
    # standard = 20, 110, 640, 580
    tensor = []
    image = PIL.ImageGrab.grab(bbox=(emulatorBox[0], emulatorBox[1], emulatorBox[2], emulatorBox[3]))
    image_gray = PIL.ImageOps.grayscale(image)
    tensor.append(np.array(image_gray))

    time.sleep(0.5)

    image = PIL.ImageGrab.grab(bbox=(emulatorBox[0], emulatorBox[1], emulatorBox[2], emulatorBox[3]))
    image_gray = PIL.ImageOps.grayscale(image)
    tensor.append(np.array(image_gray))

    time.sleep(0.5)

    image = PIL.ImageGrab.grab(bbox=(emulatorBox[0], emulatorBox[1], emulatorBox[2], emulatorBox[3]))
    image_gray = PIL.ImageOps.grayscale(image)
    tensor.append(np.array(image_gray))

    time.sleep(0.5)

    image = PIL.ImageGrab.grab(bbox=(emulatorBox[0], emulatorBox[1], emulatorBox[2], emulatorBox[3]))
    image_gray = PIL.ImageOps.grayscale(image)
    tensor.append(np.array(image_gray))

    # score standard crop = 460, 122, 585, 145
    cropped = image.crop((scoreBox[0], scoreBox[1], scoreBox[2], scoreBox[3]))  # score is here
    crop_src = 'res_crop.png'
    cropped.save(crop_src)

    pts0 = findNumbers(crop_src, 'images/0.png')
    pts1 = findNumbers(crop_src, 'images/1.png')
    pts2 = findNumbers(crop_src, 'images/2.png')
    pts3 = findNumbers(crop_src, 'images/3.png')
    pts4 = findNumbers(crop_src, 'images/4.png')
    pts5 = findNumbers(crop_src, 'images/5.png')
    pts6 = findNumbers(crop_src, 'images/6.png')
    pts7 = findNumbers(crop_src, 'images/7.png')
    pts8 = findNumbers(crop_src, 'images/8.png')
    pts9 = findNumbers(crop_src, 'images/9.png')
    pts = []
    for pt in pts0:
        pts.append(Point(0, pt))
    for pt in pts1:
        pts.append(Point(1, pt))
    for pt in pts2:
        pts.append(Point(2, pt))
    for pt in pts3:
        pts.append(Point(3, pt))
    for pt in pts4:
        pts.append(Point(4, pt))
    for pt in pts5:
        pts.append(Point(5, pt))
    for pt in pts6:
        pts.append(Point(6, pt))
    for pt in pts7:
        pts.append(Point(7, pt))
    for pt in pts8:
        pts.append(Point(8, pt))
    for pt in pts9:
        pts.append(Point(9, pt))

    pts.sort(key=lambda x: x.spot)

    score_text = ""

    for point in pts:
        score_text += str(point.number)
    return tensor, score_text


def test_image():
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("0")

    image = PIL.ImageGrab.grab(bbox=(20,110,640,580))
    # takes ~45ms

    stats = [(640-20), (580-110)]
    print(stats)

    cropped = image.crop((460, 122, 585, 145))
    # cropped.show()
    stats = [(585-460), (145-122)]
    print(stats)
    cropped.save('res_raw.png')

    img_src = 'res_raw.png'
    pts0 = findNumbers(img_src, 'images/0.png')
    pts1 = findNumbers(img_src, 'images/1.png')
    pts2 = findNumbers(img_src, 'images/2.png')
    pts3 = findNumbers(img_src, 'images/3.png')
    pts4 = findNumbers(img_src, 'images/4.png')
    pts5 = findNumbers(img_src, 'images/5.png')
    pts6 = findNumbers(img_src, 'images/6.png')
    pts7 = findNumbers(img_src, 'images/7.png')
    pts8 = findNumbers(img_src, 'images/8.png')
    pts9 = findNumbers(img_src, 'images/9.png')
    pts = []
    for pt in pts0:
        pts.append(Point(0, pt))
    for pt in pts1:
        pts.append(Point(1, pt))
    for pt in pts2:
        pts.append(Point(2, pt))
    for pt in pts3:
        pts.append(Point(3, pt))
    for pt in pts4:
        pts.append(Point(4, pt))
    for pt in pts5:
        pts.append(Point(5, pt))
    for pt in pts6:
        pts.append(Point(6, pt))
    for pt in pts7:
        pts.append(Point(7, pt))
    for pt in pts8:
        pts.append(Point(8, pt))
    for pt in pts9:
        pts.append(Point(9, pt))

    print(pts)
    pts.sort(key=lambda x: x.spot)
    print(pts)

    text = ""

    for point in pts:
        text += str(point.number)

    print("---Text---")
    print(text)
    print("---Done---")
