import cv2
import numpy as np
import os


path = 'images'
images = []
class_names = []
images_list = os.listdir(path)
orb = cv2.ORB_create(nfeatures=1000)

for image in images_list:
    img = cv2.imread(f'{path}/{image}',0)
    images.append(img)
    class_names.append(os.path.splitext(image)[0])

def findDes(images):
    des_list=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        des_list.append(des)
    return des_list
des_list = findDes(images)

def finID(img, des_list, thres = 15):

    kp2, des2 = orb.detectAndCompute(img,None)
    bf_match = cv2.BFMatcher()
    match_list= []
    finalVal = -1
    try:
        for des1 in des_list:    
            matches = bf_match.knnMatch(des1, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance< 0.75* n.distance:
                    good.append([m])
            match_list.append(len(good))
    except:
        pass
    if len(match_list)!=0:
        if max(match_list)> thres :
            finalVal = match_list.index(max(match_list))
        return finalVal  
des_list = findDes(images)

camera =  cv2.VideoCapture(0)

while True:
    success , img_cam = camera.read()
    img_color = img_cam.copy()
    img_cam = cv2.cvtColor(img_cam,cv2.COLOR_BGR2GRAY)

    id = finID(img_cam,des_list)
    if id != -1:
        cv2.putText(img_color, class_names[id], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)

    cv2.imshow('img_cam',img_color)
    cv2.waitKey(1)
