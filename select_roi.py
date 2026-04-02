import cv2

cap = cv2.VideoCapture(0)

# Select the region of interest (ROI) on the first frame

ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    exit()

# Let the user select the ROI
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")
print(f"Selected ROI: {roi}")
                    