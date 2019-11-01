# https://towardsdatascience.com/creating-ai-for-gameboy-part-2-collecting-data-from-the-screen-ccd7381a1a33
# https://towardsdatascience.com/creating-ai-for-gameboy-part-3-automating-awful-gameplay-b60fe7504e4e
# https://github.com/aaronfrederick/Fire-Emblem-AI
import time
import timeit

import cv2
import numpy as np
import pytesseract as pytesseract
import matplotlib.pyplot as plt

def subscreen(x0,y0,x1,y1, screen):
    sub_img = []
    for i in range(y0,y1,1):
        row=[]
        for j in range(x0,x1,1):
            row.append(screen[i][j])
        sub_img.append(np.array(row))
    sub_img = np.array(sub_img)
    return sub_img

import PIL.ImageGrab
import PIL.ImageOps

print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("0")
image = PIL.ImageGrab.grab(bbox=(20,110,640,580))
#takes ~45ms

stats = [(640-20), (580-110)]
print(stats)

#cropped = image.crop((460, 122, 585, 145))
cropped = image.crop((460, 122, 585, 145))
#cropped.show()
stats = [(585-460), (145-122)]
print(stats)
cropped.save('res_raw.png')

#for black text
#text = pytesseract.image_to_string(image)

#https://stackoverflow.com/questions/50951955/pytesseract-tesseractnotfound-error-tesseract-is-not-installed-or-its-not-i
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#for white text
#image.convert('L')
#conv = cropped.convert('L')
#image.show()
invert = PIL.ImageOps.invert(cropped)
#resize = PIL.ImageOps.scale(invert, 3)
#text = pytesseract.image_to_string(invert, lang='eng', config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
#text = pytesseract.image_to_string(resize, lang='eng', config="-c tessedit_char_whitelist=0123456789X")

#imgArr = np.array(invert)
#model = 'block_text_logreg.pkl'
#block_reader = np.pickle.load(open(model, 'rb'))
#a = int(block_reader.predict(invert).reshape(30 * 24 * 4).reshape(1, -1))[0]

#cv2 comparison
#https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html

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


img_src = 'res_raw.png'
pts0 = findNumbers(img_src, '0.png')
pts1 = findNumbers(img_src, '1.png')
pts2 = findNumbers(img_src, '2.png')
pts3 = findNumbers(img_src, '3.png')
pts4 = findNumbers(img_src, '4.png')
pts5 = findNumbers(img_src, '5.png')
pts6 = findNumbers(img_src, '6.png')
pts7 = findNumbers(img_src, '7.png')
pts8 = findNumbers(img_src, '8.png')
pts9 = findNumbers(img_src, '9.png')
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

#for pt in zip(*loc[::-1]):
#    cv2.rectangle(img_bw, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
#cv2.imwrite('res.png', img_bw)

print("---Text---")
#text = pytesseract.image_to_string(invert, config="--psm 6")
print(text)

print("---Done---")
#invert.show()

#using logistic regression model 'block_reader'
#padder function adds black to outline if image is too small
#stats['str'] = block_reader.predict(padder(str_img).reshape(to fit model))[0]