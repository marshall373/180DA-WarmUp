#Reference:
# kmean example from OpenCV
# Debug with ChatGPT
import cv2 as cv
import numpy as np

def get_dominant_color(roi, k=3):
    data = roi.reshape((-1, 3)).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv.kmeans(data, k, None, criteria, 5, cv.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    dom = centers[np.argmax(counts)].astype(np.uint8).tolist()
    return dom

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")
    k = 3
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        s = int(min(h, w) * 0.4)
        x1, y1 = w//2 - s//2, h//2 - s//2
        x2, y2 = x1 + s, y1 + s
        roi = frame[y1:y2, x1:x2]
        dom_bgr = get_dominant_color(roi, k)
        hsv = cv.cvtColor(np.uint8([[dom_bgr]]), cv.COLOR_BGR2HSV)[0,0]
        dom_hsv = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
        disp = frame.copy()
        cv.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
        cv.putText(disp, f"K={k} Dom BGR={dom_bgr} HSV={dom_hsv}", (10,30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv.LINE_AA)
        cv.putText(disp, f"K={k} Dom BGR={dom_bgr} HSV={dom_hsv}", (10,30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv.LINE_AA)
        cv.putText(disp, "Press 'k' to change k (2-6), 'q' to quit", (10,h-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
        cv.imshow("DominantColorSimple", disp)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('k'):
            k += 1
            if k > 6: k = 2
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
