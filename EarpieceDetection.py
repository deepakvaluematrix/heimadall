import os
import cv2
import numpy as np
import time
import mediapipe as mp
from datetime import datetime

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
class EarpieceDetection:
    def __init__(self,ear_model_file_path):
        pass
        self.ear_model_file_path=ear_model_file_path        
    
    def run(self,meeting_id,VIDEO_URL,exit_event,coll,batch_id)->dict:   
        log={"Batch_Id":batch_id,"MeetingId":meeting_id}
        log["info"]=f"Capturing VideoURL:{VIDEO_URL} for earpiece detection\n"
        cap = cv2.VideoCapture(VIDEO_URL) 
        os.makedirs("cheating_images",exist_ok=True)
        d={}
        log_li=[]
        c=0
        label_li=[]
        confidence_li=[]
        #mp_drawing = mp.solutions.drawing_utils
        #facemesh=mp.solutions.face_mesh
        #face_mesh=facemesh.FaceMesh(static_image_mode=True,refine_landmarks=True,min_tracking_confidence=0.6,min_detection_confidence=0.6)
        while True:
            t1=time.perf_counter()
            success,img=cap.read()
            if success==False:
                if len(log_li)>0 and log_li[-1]!="No frame recieved by streaming":
                    log_li.append("No frame recieved by streaming")
                    log["info"]+="No frame recieved by streaming\n"
                    c=-1
                elif len(log_li)==0:    
                    log_li.append("No frame recieved by streaming")
                    log["info"]+="No frame recieved by streaming\n"
                    c=-1
                else:
                    if exit_event.is_set():
                        c=1
                    else:
                        c=-1             
            else:
                log_li=[]   
            if c==1:
                break
            elif c==-1:
                c=0
                continue
            img = cv2.resize(img, (640,640), fx=0.4, fy=0.4)
            net=cv2.dnn.readNet(self.ear_model_file_path)
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
            net.setInput(blob)
            preds = net.forward()
            preds = preds.transpose((0, 2, 1))
            #class_ids, confs, boxes = list(), list(), list()
            classes=[]
            confidences=[]
            boxes=[]
            image_height, image_width, _ = img.shape
            x_factor = image_width / INPUT_WIDTH
            y_factor = image_height / INPUT_HEIGHT

            rows = preds[0].shape[0]

            for i in range(rows):
                row = preds[0][i]
                conf = row[4]
                classes_score = row[4:]
                _,_,_, max_idx = cv2.minMaxLoc(classes_score)
                class_id = max_idx[1]
                if (classes_score[class_id] > .25):
                    confidences.append(100*round(classes_score[class_id],2))
                    classes.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)    
            earpiece="NotDetermined"
            earpiece_confidence=0
            no_earpiece="NotDetermined"
            no_earpiece_confidence=0
            for i in range(len(boxes)): 
                if classes[i]==0:
                    if confidences[i]>earpiece_confidence:
                        earpiece=True
                        earpiece_confidence=confidences[i]   
                if classes[i]==1:
                    if confidences[i]>no_earpiece_confidence:
                        no_earpiece=True
                        no_earpiece_confidence=confidences[i]    
            if earpiece==True and no_earpiece=="NotDetermined":
                label_li.append(1)
                confidence_li.append(earpiece_confidence)
            elif no_earpiece==True and earpiece=="NotDetermined":
                label_li.append(0)
                confidence_li.append(no_earpiece_confidence) 
            elif earpiece==True and no_earpiece==True:
                if earpiece_confidence>=no_earpiece_confidence:
                    label_li.append(1)
                    confidence_li.append(earpiece_confidence)
                else:
                    label_li.append(0)
                    confidence_li.append(no_earpiece_confidence)
            else:
                pass                                                                        
            if exit_event.is_set():
                cap.release()   
                break
        d['EarpieceLabels']=label_li
        d['ConfidenceLabels']=confidence_li
        log["info"]+=f"Earpiece Detection Results:\n{d}"        
        coll.insert_one(log)
        return d   
    
    def run_ear_img(self,img):
        orientation=""
        img = cv2.resize(img, (720,720), fx=0.4, fy=0.4)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks is not None:
            img_ht,img_wd=image.shape[:2]
            mesh_points=np.array([np.multiply([p.x,p.y],[img_wd,img_ht]).astype(int) for p in results.pose_landmarks.landmark])
            nose_x=mesh_points[0][0]
        else:
            nose_x=0
            orientation="NotDetermined"    
        net=cv2.dnn.readNet(self.ear_model_file_path)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        preds = preds.transpose((0, 2, 1))
        #class_ids, confs, boxes = list(), list(), list()
        classes=[]
        confidences=[]
        boxes=[]
        image_height, image_width, _ = img.shape
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT

        rows = preds[0].shape[0]

        for i in range(rows):
            row = preds[0][i]
            conf = row[4]
            classes_score = row[4:]
            _,_,_, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            if (classes_score[class_id] > .25):
                confidences.append(classes_score[class_id])
                classes.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)    
        
        earpiece="NotDetermined"
        earpiece_boxes=[]
        earpiece_confidence=0
        no_earpiece="NotDetermined"
        no_earpiece_boxes=[]
        no_earpiece_confidence=0
    
        for i in range(len(boxes)):   
            if classes[i]==0:
                earpiece_boxes+=[boxes[i]]
                if confidences[i]>earpiece_confidence:
                    earpiece=True
                    earpiece_confidence=round(np.float64(confidences[i]),2)   

            if classes[i]==1:
                no_earpiece_boxes+=[boxes[i]]
                if confidences[i]>no_earpiece_confidence:
                    no_earpiece=True
                    no_earpiece_confidence=round(np.float64(confidences[i]),2)   
        if earpiece==True:
            x_center=int(earpiece_boxes[0][0]+earpiece_boxes[0][2]/2)
            if nose_x-x_center>0 and orientation!="NotDetermined":
                orientation+="RightEar"
            elif nose_x-x_center<0 and orientation!="NotDetermined":
                orientation+="LeftEar"    
        if no_earpiece==True:
            x_center=int(no_earpiece_boxes[0][0]+no_earpiece_boxes[0][2]/2) 
            if nose_x-x_center>0 and orientation!="NotDetermined":
                orientation+="RightEar"
            elif nose_x-x_center<0 and orientation!="NotDetermined":
                orientation+="LeftEar"   
        try:
            for (x,y,w,h) in earpiece_boxes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)     
            #cv2.putText(img, f"Status:Earpiece({round(100*earpiece_confidence,2)})", (2,20), cv2.FONT_HERSHEY_SIMPLEX, img.shape[1]//500, (0, 0, 255), 5)
        except:
            pass 
        try:   
            for (x,y,w,h) in no_earpiece_boxes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)     
            #cv2.putText(img, f"Status:NoEarpiece({round(100*no_earpiece_confidence,2)})", (2,40), cv2.FONT_HERSHEY_SIMPLEX, img.shape[1]//500, (0, 255, 0), 5)
        except:
            pass       
        #cv2.imwrite(f"cheating_images/{datetime.now().strftime('%Hh%Mm%Ss')}.jpg",img)
        return cv2.imencode('.jpg', img)[1].tostring(),{"Earpiece":earpiece,"EarpieceConfidence":earpiece_confidence,"NoEarpiece":no_earpiece,"NoEarpieceConfidence":no_earpiece_confidence,"Orientation":orientation}        
        #return img