import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("97.keras")
classes = ["employed", "unemployed"]

IMG_SIZE = 48
ZOOM_FACTOR = 3


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    center_x, center_y = w//2, h//2
    radius_x, radius_y = int(w/(2*ZOOM_FACTOR)), int(h/(2*ZOOM_FACTOR))
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y

    zoom_frame = frame[min_y:max_y, min_x:max_x]
    zoom_frame = cv2.resize(zoom_frame, (w, h))

    gray = cv2.cvtColor(zoom_frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    prob = pred[0][0]
    idx = 1 if prob > 0.5 else 0
    label = f"{classes[idx]} ({prob:.2f})"

    cv2.putText(zoom_frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("UnemploymentRecognition", zoom_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

