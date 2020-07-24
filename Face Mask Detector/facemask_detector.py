#importing the libraries
import cv2
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference

print(" Loading the model ")
model = load_pytorch_model('F:/Facemask Detector/models/model360.pth');
# anchor configuration
#feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'No Mask'}
font = cv2.FONT_HERSHEY_PLAIN
writer=None

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0 
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        font, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info

print(" Opening the video file ")
cap = cv2.VideoCapture("F:/Facemask Detector/video5.mp4")
#cap = cv2.VideoCapture(0)
while True:
    grabbed,frame= cap.read() 
    if not grabbed:
        break;
    height,width,channels = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)  
    if (height<=1280 & width<=720):
        frame = cv2.resize(frame,(1280,720))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inference(frame,0.5,iou_thresh=0.5,target_shape=(360, 360),draw_result=True,show_result=False)
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,255,255),1)
    if height>1280 & width>720:
        frame = cv2.resize(frame,(1280,720))
        cv2.imshow("Detection",frame[:, :, ::-1])
    else:
        cv2.imshow("Detection",frame[:, :, ::-1]) 
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output_facemask.avi", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
    #cv2.imshow('image', frame[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break;

print(" Cleaning Up ")   
cap.release()  
writer.release()
cv2.destroyAllWindows()