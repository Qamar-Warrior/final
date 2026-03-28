import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

results = {}
best_plate = {}  # car_id -> best (text, score) seen so far
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
lp_detector = YOLO('/home/jaloliddin/License Plate/best.pt')

cap = cv2.VideoCapture('carp.mp4')
vehicles = [2, 3, 5, 7] # car, motorcycle, bus, truck

frame_nmr = -1
while True:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret: break

    results[frame_nmr] = {}
    
    # 1. Detect Vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for d in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = d
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # 2. Track Vehicles
    track_ids = mot_tracker.update(np.asarray(detections_) if detections_ else np.empty((0, 5)))

    # 3. Detect & Process Plates
    license_plates = lp_detector(frame)[0]
    for lp in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp

        # Assign plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)

        if car_id != -1:
            # Crop & OCR
            lp_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            lp_text, lp_text_score = read_license_plate(lp_crop)

            if lp_text is not None:
                prev_score = best_plate.get(car_id, (None, 0.0))[1]
                if lp_text_score > prev_score:
                    best_plate[car_id] = (lp_text, lp_text_score)

                # Always use best known text for this car
                final_text, final_score = best_plate[car_id]
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {'bbox': [x1, y1, x2, y2], 'text': final_text,
                                      'bbox_score': score, 'text_score': final_score}}
                
                # --- DEBUG VISUALIZATION ---
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"ID:{int(car_id)} {final_text}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('ALPR Uzbekistan Debug', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

write_csv(results, './test.csv')
cap.release()
cv2.destroyAllWindows()