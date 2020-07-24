import cv2
import numpy as np
import time

classname = []
def get_count(boxes, class_names):
	total_count = 0 # total vechiles present in the image
	#dict_count = {} # dictionary with count of each distinct vehicles detected
	for i in range(len(boxes)):
		#class_name = class_names[i]
		# print(i,".",class_name)
		if(class_names == 'person'):
			total_count += 1
			#dict_count[class_name] = dict_count.get(class_name,0) + 1

	return total_count

#Load YOLO
net = cv2.dnn.readNet("F:/objectdetection/YOLO FILES/yolov3.weights","F:/objectdetection/YOLO FILES/yolov3.cfg") # Original yolov3
classes = []
with open("F:/objectdetection/model/coco_classes.txt","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

#loading video
cap=cv2.VideoCapture("F:/objectdetection/people_video.mp4")
#out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1280,720))
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0
person=0
writer=None

while True:
    grabbed,frame= cap.read() # 
    frame_id+=1
    if not grabbed:
        break;
    height,width,channels = frame.shape
    if (height<=1280 & width<=720):
        frame = cv2.resize(frame,(1280,720))
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320           
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])
    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    location=[]
    for out in outs:
        for detection in out:
            #start = timer()
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                #object detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                #rectangle co-ordinates
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object that was detected
                #cv2.line()
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    counter=0
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]  
            center_x=(2*x+w)/2
            center_y=(2*y+h)/2
            label = str(classes[class_ids[i]])
            if label=='person':
                confidence= confidences[i]
                color = colors[class_ids[i]]
                total_person = get_count(boxes, label)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            #cv2.putText(frame,label+" "+str(round(confidence,2))+" SPEED: "+str(round(speeds))+" km/h",(x,y+30),font,1,(0,255,255),2)
                cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(0,255,255),2)
    fps = cap.get(cv2.CAP_PROP_FPS)        
    #elapsed_time = time.time() - starting_time
    #fps=frame_id/elapsed_time      
    person = total_person
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,255,255),1)
    cv2.putText(frame," COUNTER : "+str(person),(10,300),font,3,(0,255,255),3)
    #out.write(frame)
    if height>1280 & width>720:
        cv2.imshow("Vehicle_Detection",cv2.resize(frame,(1280,720)))
    else:
        cv2.imshow("Vehicle_Detection",frame)   
    #key = cv2.waitKey(1)#wait 1ms the loop will start again and we will process the next frame
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output_people.avi", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
print("[INFO] Total Person in the Video: {:.2f}".format(person))
print("Cleaning Up....")   
cap.release()  
writer.release()  
cv2.destroyAllWindows()
