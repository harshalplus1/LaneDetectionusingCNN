import tensorflow as tf
import numpy as np
import cv2 as cv

samplevid=cv.VideoCapture(r"C:\Users\hkshi\Downloads\Lane detect test data.mp4")
size=(80,160)
model=tf.keras.models.load_model(r"C:\Users\hkshi\OneDrive\Desktop\LaneDetectionmodel.h5")
class Lanes():
    def __init__(self):
        self.lanes=[]
        self.avglanes=[]

ln=Lanes()
while(True):
    ret,frame1=samplevid.read()
    frame=frame1
    frame=cv.resize(frame,(160,80),cv.INTER_NEAREST)
    farray=np.array(frame)
    farray=farray[None,:,:,:]
    y_pred=model.predict(farray)[0]*255
    ln.lanes.append(y_pred)
    if len(ln.lanes)>7:
        ln.lanes=ln.lanes[1:]
    ln.avglanes= np.mean(np.array([pred for pred in ln.lanes]),axis=0)
    blanks=np.zeros_like(ln.avglanes)
    drawline=np.dstack((blanks,y_pred,blanks))
    img1=np.array(cv.resize(drawline,(1280,720)),dtype=np.uint8)
    result=cv.addWeighted(frame1,1,img1,1,0)
    blur=cv.blur(result,(5,5))
    cv.imshow("Result",blur)
    if cv.waitKey(1) & 0xFF==ord("s"):
         break
samplevid.release()
cv.waitKey(0)

    