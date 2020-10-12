import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

path="D:/COMPLETE DATA SCIENCE AND MACHINE LEARNING/19-Open Cv/Image Samples/" # ye vo path haii jha mere pics k saare samples haii
files_list=[files for files in listdir(path) if isfile(join(path,files))] # extracting the images
# print(files_list)
# print(join(path,files_list[0]))
X_train,Labels=[],[]

# splitting the data into training data
for i,files in enumerate(files_list):
    image_path=path+files_list[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    X_train.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels=np.asarray(Labels,dtype=np.int32)
model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(X_train),np.asarray(Labels))
print("MODEL CREATED SUCCESSFULLY......!!!!")

face_classifier=cv2.CascadeClassifier("C:/Users/satvik sharma/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

def Face_Detector(img,size=0.5):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray_img,1.3,5)

    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)#(jispe rectangle draw krna haii,starting points,end points,color,line width)
        roi=img[y:y+h,x:x+w]# reason of interest
        roi=cv2.resize(roi,(200,200))
    return img,roi

cap=cv2.VideoCapture(0)

while True:

    ret,frame=cap.read()
    image,face=Face_Detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)

        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))
            display_string=str(confidence)+"% Confidence it is Satvik"
            cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,0),2)
        
        if confidence>75:
            cv2.putText(image,"UN-LOCKED",(200,400),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
            cv2.imshow("FACE RECOGNIZER BY S@TVIK",image)
        else:
            cv2.putText(image,"LOCKED",(200,400),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
            cv2.imshow("FACE RECOGNIZER BY S@TVIK",image)

    except:
        cv2.putText(image,"FACE NOT FOUND........!!!",(100,120),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
        cv2.imshow("FACE RECOGNIZER BY S@TVIK",image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()

# python faceRecognition_2.py
