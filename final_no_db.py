import cv2
import time
import numpy as np
import tomllib
from gpiozero import RGBLED, Button, LED
from rpi_hardware_pwm import HardwarePWM
from garbage_detect import classify

try:
    from voice_ai_tts import tts
except Exception:
    print("Using alternative TTS due to import error.")
    from tts import tts

# ==========================================
# 1. CONFIGURATION & PINOUT
# ==========================================
MOTION_THRESHOLD = 10000
ROI = (80, 38, 888, 901)  # x, y, w, h
SERVO_FREQUENCY = 50
CLOSED_CYCLE = 7.3
OPEN_CYCLE = 2.5

# ==========================================
# 2. UTILITIES
# ==========================================

def load_pinout(path="pinout.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


def crop_to_roi(frame):
    return frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]


def speak(message, block=True):
    print(message)
    try:
        tts(message, block=block)
    except Exception as exc:
        print(f"TTS error: {exc}")

# ==========================================
# 3. HARDWARE INITIALIZATION
# ==========================================
print("Initializing AI, voice, and hardware...")

CATEGORIES = load_pinout()

class Bin:
    def __init__(self, name, config):
        self.name = name
        self.led = RGBLED(red=config["led"][0], green=config["led"][1], blue=config["led"][2])
        self.button = Button(config["button"], pull_up=True) # bounce time seems to cause missed presses, so omitting for now
        self.servo = HardwarePWM(pwm_channel=config["servo"], hz=SERVO_FREQUENCY, chip=0)
        self.servo.start(CLOSED_CYCLE)
        time.sleep(0.2)
        self.servo.change_duty_cycle(0)

    def move(self, duty_cycle):
        self.servo.change_duty_cycle(duty_cycle)
        time.sleep(0.8)
        self.servo.change_duty_cycle(0)

    def reset(self):
        self.led.off()

    def shutdown(self):
        try:
            self.servo.change_duty_cycle(0)
            self.servo.stop()
        except Exception:
            pass


SCAN_LIGHT = LED(CATEGORIES.pop("scan_led"))
bins = {name: Bin(name, config) for name, config in CATEGORIES.items()}

print(f"Loaded categories: {', '.join(bins.keys())}")


def move_hatch(bin_obj, duty_cycle):
    bin_obj.move(duty_cycle)


def flash_red(bin_name):
    bins[bin_name].led.color = (1, 0, 0)
    time.sleep(0.5)
    bins[bin_name].led.off()


def reset_leds():
    SCAN_LIGHT.off()
    for b in bins.values():
        b.reset()


# ==========================================
# 4. MAIN LOOP
# ==========================================

cap = cv2.VideoCapture(0)
time.sleep(2)

try:
    # speak("Trash sorting system is ready. Please place the item in the scanning area.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera not found")

    roi_frame = crop_to_roi(frame)
    prev_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    counter = 0

    while True:
        reset_leds()
        print("\n--- Waiting for item to be placed ---")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            counter += 1

            if counter % 30 == 0:
                SCAN_LIGHT.toggle()

            roi_frame = crop_to_roi(frame)
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            diff = cv2.absdiff(prev_gray, gray)
            motion_score = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1])

            if motion_score > MOTION_THRESHOLD:
                print("Motion detected! Settling...")
                SCAN_LIGHT.on()

                # 1. Wait for physical motion to stop and camera exposure to adjust
                # (Do NOT read from the camera during this sleep)
                time.sleep(1.5) 
                
                # 2. Rapidly flush the camera's internal buffer 
                # cap.grab() is much faster than cap.read() because it throws away the 
                # frame without spending CPU time decoding the image data.
                for _ in range(10):  
                    cap.grab()

                print("Capturing image for classification...")
                ret, settled_frame = cap.read()
                snapshot = crop_to_roi(settled_frame)
                break

            prev_gray = gray
            cv2.waitKey(10)

        cv2.imwrite("snapshot.jpg", snapshot)
        _, buffer = cv2.imencode('.jpg', snapshot)
        image_bytes = buffer.tobytes()
        classification = classify(image_bytes, 'image/jpeg')
        actual_cat = classification.get("Category", "recycling").lower()
        explanation = classification.get("Explanation", "")
        description = classification.get("Object Description", "")

        if actual_cat not in bins:
            print(f"AI returned unsupported category '{actual_cat}'. Defaulting to garbage.")
            actual_cat = "garbage"
        else:
            message = f"AI identifies {description}. It belongs in {actual_cat}."
            print(message)

        SCAN_LIGHT.off()
        
        guesses = []
        available_options = list(bins.keys())
        last_blink = time.time()
        blue_state = False

        # speak("Please make your guess by pressing the button for the bin.")
        print("Waiting for student guess...")

        while True:
            if time.time() - last_blink > 0.5:
                blue_state = not blue_state
                for opt in available_options:
                    bins[opt].led.color = (0, 0, 1) if blue_state else (0, 0, 0)
                last_blink = time.time()

            user_guess = None
            for name, b_unit in bins.items():
                if b_unit.button.is_pressed:
                    user_guess = name
                    break

            if user_guess:
                for b in bins.values():
                    b.led.off()

                if user_guess not in guesses:
                    guesses.append(user_guess)

                if user_guess == actual_cat:
                    bins[user_guess].led.color = (0, 1, 0)
                    break
                else:
                    flash_red(user_guess)
                    if user_guess in available_options:
                        available_options.remove(user_guess)

            cv2.waitKey(10)

        # if explanation:
        #     speak(explanation, block=False)
        move_hatch(bins[actual_cat], OPEN_CYCLE)
        time.sleep(4.0)
        move_hatch(bins[actual_cat], CLOSED_CYCLE)

        ret, fresh = cap.read()
        if not ret:
            continue
        roi_fresh = crop_to_roi(fresh)
        prev_gray = cv2.cvtColor(roi_fresh, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

except KeyboardInterrupt:
    print("\nShutting down...")
    # speak("Shutting down the trash sorting system.")
finally:
    reset_leds()
    for b in bins.values():
        b.shutdown()
    cap.release()
    cv2.destroyAllWindows()
