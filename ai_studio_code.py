import cv2
from ultralytics import YOLOWorld
import time

# 1. Load the Medium Model (Best for Pi 5 16GB)
model = YOLOWorld('yolov8m-worldv2.pt') 

# 2. REFINED INCLUSIVE CLASS LIST
# We use very descriptive language to help the AI "anchor" onto the objects.
CLASSES = [
    "plastic bottle with cap",          # Top-half bottle fragment
    "orange or clear plastic bottle",   # Mini Fanta / Colored bottles
    "empty plastic bottle no cap",      # Capless bottles
    "metallic soda can",                # Metal cans
    "cardboard box",                    # Boxes
    "yellow banana peel with spots",    # Organic (Specific texture)
    "orange peel or citrus skin",       # Organic (Specific color)
    "white crumpled paper",             # Paper
    "snack wrapper"                     # Wrappers
]
model.set_classes(CLASSES)

# 3. MAPPING TO BINS
BIN_MAP = {
    "PLASTIC": ["plastic", "wrapper"],
    "METAL": ["can", "metallic"],
    "PAPER/CARD": ["cardboard", "box", "paper"],
    "ORGANIC": ["peel", "skin", "citrus", "banana"]
}

# BGR Colors
COLORS = {
    "PLASTIC": (255, 0, 0),    # Blue
    "METAL": (0, 0, 255),      # Red
    "PAPER/CARD": (0, 255, 255),# Yellow
    "ORGANIC": (0, 255, 0),     # Green (Compost)
    "OTHER": (150, 150, 150)
}

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

print("Pi 5 Full-Spectrum Trash Monitor is starting...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape

    # 4. BALANCED INFERENCE
    # iou=0.2 is very strict - it prevents 'ghost' boxes from overlapping
    # agnostic_nms=True forces the different prompts to fight for the same space
    results = model.predict(frame, imgsz=640, conf=0.15, iou=0.2, agnostic_nms=True, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            item_name = model.names[cls_id].lower()

            # --- DYNAMIC SENSITIVITY ---
            # Metal is high-contrast, demand 40%. 
            # Organic/Plastic can be messy, allow 15%.
            required_conf = 0.15
            if "can" in item_name or "metallic" in item_name:
                required_conf = 0.40
            
            if conf < required_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Filter full-screen glitches
            if (x2-x1)*(y2-y1) > (h*w*0.8): continue 

            # Map to Bin
            bin_name = "OTHER"
            for b_name, keywords in BIN_MAP.items():
                if any(key in item_name for key in keywords):
                    bin_name = b_name
                    break
            
            color = COLORS.get(bin_name, (150, 150, 150))

            # 5. DRAWING
            # Draw a thick box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label with a solid background for readability
            label_text = f"{bin_name} {int(conf*100)}%"
            t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1-25), (x1+t_size[0]+10, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Trash Sorting Station", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()