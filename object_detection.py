from ultralytics import YOLO
import numpy as np
import cv2

cap = cv2.VideoCapture("/content/test_footage.mp4")

model = YOLO("/content/yolov8x-oiv7.pt")
labels = model.names
colors = np.random.uniform(0, 255, size=(len(labels), 3))

video = cv2.VideoWriter("/content/output.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))

while True:
    success, img = cap.read()
    
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f"\nFrame {currentFrame} / {totalFrames}")

    # height, width, _ = img.shape
    if(not success):
        break

    results = model(img, stream=True)
    for res in results:
        boxes = res.boxes.data
        # print(boxes.data)
        for box in boxes:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

            label = labels[int(box[-1])] + "(" + str(round(100 * float(box[-2]), 2)) + "%)"
            color = colors[int(box[-1])]

            lw = max(round(sum(img.shape) / 2 * 0.003), 2)
            cv2.rectangle(img, p1, p2, color, lw)

            ft = max(lw - 1, 1)
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=ft)[0]

            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(img, p1, p2, color, -1)

            cv2.putText(
                    img,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    (255, 255, 255),
                    thickness=ft,
                    lineType=cv2.LINE_AA
            )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        video.write(img)

            # cv2.imshow("Object Detection", img)
            # cv2.waitKey(1)
            # if cv2.waitKey(1000) & 0xFF == ord('q'):
            #      break
            # cap.release()
            # cv2.destroyAllWindows()