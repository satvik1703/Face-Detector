import cv2
# Sabse pehle mein chahunga ki meri machine face ko pehchane for tht i import cascade classifier
face_classifier=cv2.CascadeClassifier("C:/Users/satvik sharma/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

def Face_Extractor(img):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # covert the image into gray image
    faces=face_classifier.detectMultiScale(gray_img,1.3,5)# classifies the face
    
    if faces is(): # agar face nhi hota toh none return krega
        return None
    for(x,y,w,h) in faces:# agar image hoti haii toh coordiantes de dega
        cropped_img=img[y:y+h,x:x+w]
    return cropped_img

cap=cv2.VideoCapture(0) # on the camera
count=1

while True:
    ret,frame=cap.read()
    if Face_Extractor(frame) is not None: # face extractor function ne jo image crop krke di haii
        count+=1
        face=cv2.resize(Face_Extractor(frame),(200,200)) # we resize tht image
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)# covert it into gray color

        Folder_path="D:/COMPLETE DATA SCIENCE AND MACHINE LEARNING/19-Open Cv/Image Samples/user"+str(count)+".jpg" # ye vo folder haii jha saare samples save honge.
        cv2.imwrite(Folder_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("FACE CROPPER",face)
    else:
        print("IMAGE NOT FOUND!!!!")
        pass
    
    if cv2.waitKey(1)==13 or count==100:
        break
    
cap.release()
cv2.destroyAllWindows()
print("SAMPLES COLLECTED SUCCESSFULLY.....!!!")

