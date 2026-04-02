import cv2
import time
import numpy as np
from gpiozero import RGBLED, Button, Servo, LED
from ultralytics import YOLOWorld

# ==========================================
# 1. CONFIGURATION & CALIBRATION
# ==========================================
MOTION_THRESHOLD = 10000 
ROI = (80, 38, 888, 901) # x, y, width, height

# Servo Timing for MG90S on Pi 5
WIDE_MIN = 0.5 / 1000 
WIDE_MAX = 2.5 / 1000
VAL_CLOSED = 0.88  # 10 Degrees (Right/Home)
VAL_OPEN   = 0.0   # 90 Degrees (Middle/Half-Open)

# ==========================================
# 2. HARDWARE INITIALIZATION
# ==========================================
print("Initializing AI and Hardware...")

# Load the High-Accuracy Medium Model
model = YOLOWorld('yolov8m-worldv2.pt') 
CLASSES = [
    "plastic bottle", "metal can", "cardboard box", "paper sheet",
    "banana peel", "orange peel", "apple core", "food waste",
    "snack wrapper", "black plastic container"
]
model.set_classes(CLASSES)

# Dedicated Scanning Light
scan_light = LED(14)

class BinUnit:
    def __init__(self, name, r, g, b, btn, servo_pin):
        self.name = name
        self.led = RGBLED(red=r, green=g, blue=b)
        self.button = Button(btn)
        self.servo = Servo(servo_pin, min_pulse_width=WIDE_MIN, max_pulse_width=WIDE_MAX)
        
        # Zero-Jitter Startup
        self.servo.value = VAL_CLOSED
        time.sleep(0.5)
        self.servo.detach()

# BINS (Mapped to your specific pinout)
BINS = {
    "garbage":   BinUnit("Garbage", 3, 4, 17, 19, 27),
    "recycling": BinUnit("Recycling", 22, 10, 9, 11, 18),
    "compost":   BinUnit("Compost", 5, 6, 13, 15, 23)
}

# ==========================================
# 3. LOGIC FUNCTIONS
# ==========================================

def get_category(label):
    label = label.lower()
    # Halton Region Rule: Black plastic is always Garbage
    if "black plastic" in label: return "garbage"
    if any(x in label for x in ["plastic", "can", "cardboard", "box", "paper"]):
        return "recycling"
    if any(x in label for x in ["peel", "core", "food", "orange", "banana"]):
        return "compost"
    return "garbage"

def jitter_free_move(unit, target):
    """Moves servo and kills signal immediately for Arduino-level stability."""
    unit.servo.value = target
    time.sleep(0.8) # Wait for movement
    unit.servo.detach()

def reset_all_leds():
    scan_light.off()
    for b in BINS.values():
        b.led.off()

# ==========================================
# 4. MAIN PROGRAM LOOP
# ==========================================
cap = cv2.VideoCapture(0)
time.sleep(2) # Camera warmup

try:
    # Setup baseline for motion detection
    ret, frame = cap.read()
    if not ret: raise Exception("Camera not found")
    
    roi_frame = frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
    prev_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while True:
        reset_all_leds()
        print("\n--- READY: Place item in scanning area ---")
        
        # STEP 1: Motion Detection in ROI
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            roi_frame = frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])
            
            if motion_score > MOTION_THRESHOLD:
                print(f"Motion detected! (Score: {motion_score})")
                scan_light.on()
                time.sleep(2.0) # Wait for hand to leave and object to settle
                
                # Take final snapshot for AI
                ret, frame = cap.read()
                snapshot = frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                break
                
            prev_gray = gray
            cv2.waitKey(10)

        # STEP 2: AI Inference
        results = model.predict(snapshot, conf=0.2, imgsz=640, iou=0.2, agnostic_nms=True, verbose=False)
        
        if len(results[0].boxes) > 0:
            item_name = model.names[int(results[0].boxes[0].cls[0])]
            correct_bin = get_category(item_name)
            print(f"AI Identify: {item_name} -> Destination: {correct_bin}")
        else:
            print("AI unsure. Defaulting to Garbage for safety.")
            correct_bin = "garbage"

        # STEP 3: Educational Guessing Game
        print("Waiting for student guess...")
        guesses = []
        available_options = list(BINS.keys())
        last_blink = time.time()
        blue_on = False

        while True:
            # Blink active bin choices Blue
            if time.time() - last_blink > 0.5:
                blue_on = not blue_on
                for opt in available_options:
                    BINS[opt].led.color = (0, 0, 1) if blue_on else (0, 0, 0)
                last_blink = time.time()

            # Check for button press
            user_guess = None
            for name, unit in BINS.items():
                if unit.button.is_pressed:
                    user_guess = name
                    break
            
            if user_guess:
                for b in BINS.values(): b.led.off() # Stop all blinks
                
                if user_guess not in guesses: guesses.append(user_guess)

                if user_guess == correct_bin:
                    print("CORRECT! Green light.")
                    BINS[user_guess].led.color = (0, 1, 0)
                    break
                else:
                    print(f"WRONG: {user_guess} is not correct.")
                    BINS[user_guess].led.color = (1, 0, 0)
                    time.sleep(0.8)
                    BINS[user_guess].led.off()
                    if user_guess in available_options:
                        available_options.remove(user_guess) # Stop blinking this one
            
            cv2.waitKey(10)

        # STEP 4: Physical Disposal
        print(f"Opening {correct_bin} hatch...")
        target_unit = BINS[correct_bin]
        jitter_free_move(target_unit, VAL_OPEN)
        
        time.sleep(4.0) # Time for item to drop
        
        print("Closing hatch.")
        jitter_free_move(target_unit, VAL_CLOSED)
        
        # Reset baseline for next item
        ret, fresh = cap.read()
        roi_fresh = fresh[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
        prev_gray = cv2.cvtColor(roi_fresh, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        print("System Ready.")

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    reset_all_leds()
    for b in BINS.values():
        b.servo.detach()
    cap.release()
    cv2.destroyAllWindows()