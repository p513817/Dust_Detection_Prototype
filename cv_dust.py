import cv2
import numpy as np
import cv2
import imutils
import random

def find_circle(img):

    img_draw = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)    

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 0.001, 10)
    print(circles)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img_draw, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img_draw, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    return img_draw

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)
 
    out = sum(contour_list)
    return out

def find_dust(img, min=5, max=15, thred=150, max_area=1400):
    img_draw  = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    erode = cv2.erode(blurred, (3, 3), iterations=3)
    dilate = cv2.dilate(erode, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations = 2)
    # edged = cv2.Cnny(dilate, thred_min, thred_max)#, 3)            
    ret, out = cv2.threshold(dilate, thred, 255, cv2.THRESH_BINARY)

    cnts,_ = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    idx = 0
    sth = 0

    if len(cnts) > 0:

        for idx, cnt in enumerate(cnts):    
            
            if cv2.contourArea(cnt) >= max_area:
                # print(cv2.contourArea(cnt))
                cv2.drawContours(img_draw, cnts, idx, (0, 0, 255), 2)
                sth = sth +1
            else:
                peri = cv2.arcLength(cnt, True) 
    
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)           # 多邊形

                if len(approx) > 5:
                    idx = idx+1
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    
                    # if radius > min and radius < max:

                    center = (int(x),int(y))
                    radius = int(radius)
                    cv2.circle(img_draw, center,radius,(0,255,0), 2)
    
    return img_draw, idx, sth

def img_process(frame):

    x0, y0 = (0, 160)
    x1, y1 = (420, 660)
    frame = frame[y0:y1, x0:x1]

    height, width, _ = frame.shape
    cut_edge = int((width-height)/2)
    new_frame = frame[:, cut_edge:width-cut_edge]

    frame = cv2.resize(new_frame, (480, 480))

    return frame

def stream():
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while(True):
        ret, frame = cap.read()

        frame = img_process(frame)

        if ret:
            new, idx , sth= find_dust(frame, min=5, max=30)

            cv2.imshow('asd', new)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()



if __name__ == "__main__":

    # label = ['clean', 'dust', 'other']
    # count = 0
    

    # while(True):

    #     img_idx = random.randint(0, 200)    
    #     # label_idx = random.randint(0, 2)
    #     label_idx = count%3
        
    #     img = cv2.imread(f'./data/train/{label[label_idx]}/{img_idx}.jpg')

    #     res, idx = find_dust(img)

    #     cv2.imshow('img', res)
    #     key = cv2.waitKey(0)
    #     if key is ord('q'):
    #         break
    #     elif key:
    #         pass
    #     count = count + 1
    # cv2.destroyAllWindows()

    stream()