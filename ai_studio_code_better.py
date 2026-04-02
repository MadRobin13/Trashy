import cv2
import time
import numpy as np
from gpiozero import RGBLED, Button, Servo, LED
from ultralytics import YOLOWorld

# ==========================================
# 1. CONFIGURATION & PINOUT (Your Setup)
# ==========================================
MOTION_THRESHOLD = 10000 
ROI = (80, 38, 888, 901) # x, y, w, h

# Servo Calibration (From our tests)
WIDE_MIN = 0.5 / 1000 
WIDE_MAX = 2.5 / 1000
VAL_CLOSED = 0.88  # 10 Degrees (Right)
VAL_OPEN   = 0.0   # 90 Degrees (Middle)

# Scan Area Light
SCAN_LIGHT = LED(14) 

# Bin Hardware Mapping
# Note: Servos moved to unique pins to avoid LED conflicts
CATEGORIES = {
    "garbage": {
        "red_led": 3, "green_led": 4, "blue_led": 17, 
        "button": 19, "servo_pin": 27 
    },
    "recycling": {
        "red_led": 22, "green_led": 10, "blue_led": 9, 
        "button": 11, "servo_pin": 18
    },
    "compost": {
        "red_led": 5, "green_led": 6, "blue_led": 13, 
        "button": 15, "servo_pin": 23
    }
}

# ==========================================
# 2. INITIALIZATION
# ==========================================
print("Initializing AI and Hardware...")

# Load Vision Model
model = YOLOWorld('yolov8m-worldv2.pt') 
CLASSES = [
    "plastic bottle", "metal can", "cardboard box", "paper sheet",
    "banana peel", "orange peel", "apple core", "food waste",
    "snack wrapper", "black plastic container"
]
model.set_classes(CLASSES)

# Initialize Bin Objects
class Bin:
    def __init__(self, pins):
        self.led = RGBLED(red=pins["red_led"], green=pins["green_led"], blue=pins["blue_led"])
        self.button = Button(pins["button"])
        self.servo = Servo(pins["servo_pin"], min_pulse_width=WIDE_MIN, max_pulse_width=WIDE_MAX)
        # Set to home and detach to stop jitter
        self.servo.value = VAL_CLOSED
        time.sleep(0.5)
        self.servo.detach()

bins = {name: Bin(p) for name, p in CATEGORIES.items()}

def get_category_from_label(label):
    label = label.lower()
    if "black plastic" in label: return "garbage" # Halton Rule
    if any(x in label for x in ["plastic", "can", "cardboard", "box", "paper"]):
        return "recycling"
    if any(x in label for x in ["peel", "core", "food", "waste"]):
        return "compost"
    return "garbage"

# Initialize Camera
cap = cv2.VideoCapture(0)
time.sleep(2) 

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def move_hatch(bin_obj, target):
    bin_obj.servo.value = target
    time.sleep(0.8)
    bin_obj.servo.detach()

def flash_red(bin_name):
    bins[bin_name].led.color = (1, 0, 0)
    time.sleep(0.5)
    bins[bin_name].led.off()

def reset_leds():
    SCAN_LIGHT.off()
    for b in bins.values():
        b.led.off()

# ==========================================
# 4. MAIN LOOP
# ==========================================

try:
    # Capture baseline frame for motion detection
    ret, frame = cap.read()
    roi_frame = frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
    prev_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while True:
        reset_leds()
        print("\n--- Waiting for item to be placed ---")
        
        # STEP 1: Motion Detection Loop
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            roi_frame = frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            diff = cv2.absdiff(prev_gray, gray)
            motion_score = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])
            
            if motion_score > MOTION_THRESHOLD:
                print("Motion detected! Settling...")
                SCAN_LIGHT.on() # Turn on scan light while waiting
                time.sleep(2.0) # Settle camera
                ret, frame = cap.read() # Final snapshot
                snapshot = frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                break
                
            prev_gray = gray
            cv2.waitKey(10)

        # STEP 2: AI Classification
        results = model.predict(snapshot, conf=0.2, imgsz=640, verbose=False)
        
        if len(results[0].boxes) > 0:
            item_name = model.names[int(results[0].boxes[0].cls[0])]
            actual_cat = get_category_from_label(item_name)
            print(f"AI identifies: {item_name} -> Destination: {actual_cat}")
        else:
            print("AI confused. Defaulting to Garbage.")
            actual_cat = "garbage"

        # STEP 3: Guessing Game Loop
        guesses = []
        available_options = list(CATEGORIES.keys())
        last_blink = time.time()
        blue_state = False
        
        print("Waiting for student guess...")

        while True:
            # Non-blocking Blue Blink
            if time.time() - last_blink > 0.5:
                blue_state = not blue_state
                for opt in available_options:
                    bins[opt].led.color = (0, 0, 1) if blue_state else (0, 0, 0)
                last_blink = time.time()

            # Check for button press
            user_guess = None
            for name, b_unit in bins.items():
                if b_unit.button.is_pressed:
                    user_guess = name
                    break
            
            if user_guess:
                # Stop all blinking blue
                for b in bins.values(): b.led.off()
                
                if user_guess not in guesses: guesses.append(user_guess)

                if user_guess == actual_cat:
                    print("CORRECT!")
                    bins[user_guess].led.color = (0, 1, 0) # Solid Green
                    break
                else:
                    print(f"WRONG: {user_guess} is incorrect.")
                    flash_red(user_guess)
                    if user_guess in available_options:
                        available_options.remove(user_guess)
            
            cv2.waitKey(10)

        # STEP 4: Physical Action
        print(f"Opening {actual_cat} hatch...")
        move_hatch(bins[actual_cat], VAL_OPEN)
        
        time.sleep(4.0) # Hold open
        
        print("Closing hatch.")
        move_hatch(bins[actual_cat], VAL_CLOSED)
        
        # Reset baseline for next motion detection
        ret, fresh = cap.read()
        roi_fresh = fresh[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
        prev_gray = cv2.cvtColor(roi_fresh, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    reset_leds()