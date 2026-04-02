import cv2
import pigpio
import time
import requests
import json
import base64
from openai import OpenAI

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================

# API Configuration
OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your actual key
REPORT_API_URL = "https://your-api-endpoint.com/report"

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Hardware Pins (BCM numbering)
YELLOW_WAIT_LED = 14

# Define pins for each category
# Added blue_led pins (using 5, 6, 13 as examples)
CATEGORIES = {
    "garbage":   {"red_led": 3,  "green_led": 4,  "blue_led": 17,  "servo": 4,  "button": 15},
    "recycling": {"red_led": 17, "green_led": 27, "blue_led": 22,  "servo": 22, "button": 18},
    "compost":   {"red_led": 10, "green_led": 9,  "blue_led": 13, "servo": 11, "button": 23}
}

# Servo states (Pulse widths: 500-2500. 1500 is usually center)
SERVO_CLOSED = 1000
SERVO_OPEN = 2000

# Motion detection threshold
MOTION_THRESHOLD = 10000 

# Region of interest (x, y, width, height) to focus on key area only
ROI = (80, 38, 888, 901)

# ==========================================
# INITIALIZATION
# ==========================================

print("Initializing hardware...")
pi = pigpio.pi()
if not pi.connected:
    print("Failed to connect to pigpio daemon. Did you run 'sudo pigpiod'?")
    exit()

# Setup Yellow LED
pi.set_mode(YELLOW_WAIT_LED, pigpio.OUTPUT)
pi.write(YELLOW_WAIT_LED, 0)

# Setup Category Pins
for cat, pins in CATEGORIES.items():
    # LEDs
    pi.set_mode(pins["red_led"], pigpio.OUTPUT)
    pi.write(pins["red_led"], 0)
    pi.set_mode(pins["green_led"], pigpio.OUTPUT)
    pi.write(pins["green_led"], 0)
    pi.set_mode(pins["blue_led"], pigpio.OUTPUT)
    pi.write(pins["blue_led"], 0)
    
    # Servos
    pi.set_mode(pins["servo"], pigpio.OUTPUT)
    pi.set_servo_pulsewidth(pins["servo"], SERVO_CLOSED)
    
    # Buttons (using pull-up resistors)
    pi.set_mode(pins["button"], pigpio.INPUT)
    pi.set_pull_up_down(pins["button"], pigpio.PUD_UP)

# Initialize Camera
cap = cv2.VideoCapture(0)
time.sleep(2) # Allow camera to warm up

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_pressed_button():
    """Checks if any button is currently pressed. Returns category name or None."""
    for cat, pins in CATEGORIES.items():
        if pi.read(pins["button"]) == 0: # 0 means pressed due to Pull-Up
            return cat
    return None

def flash_red(category):
    """Flashes the red LED for a specific category"""
    pin = CATEGORIES[category]["red_led"]
    pi.write(pin, 1)
    time.sleep(0.5)
    pi.write(pin, 0)

def settle_camera(wait_time=2.0):
    """Read frames for a short period to clear the camera buffer after motion."""
    start = time.time()
    while time.time() - start < wait_time:
        ret, _ = cap.read()
        if not ret:
            continue
        time.sleep(0.01)

def classify_image_with_ai(frame):
    """Encodes the frame and sends it to OpenAI's Vision API."""
    print("Encoding image and sending to OpenAI...")

    # Convert OpenCV frame to JPEG, then to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {
                    "role": "system",
                    "content": "You are an automated waste-sorting assistant. Look at the item in the image and classify it into exactly one of these three categories: 'garbage', 'recycling', or 'compost'. Reply ONLY with the category word in lowercase. Do not include punctuation."
                },
                {
                    "role": "user",
                    "content":[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low" 
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_schema", "json_schema": {
                "type": "object",
                "properties": {
                    "object_description": {
                        "type": "string",
                        "description": "A brief description of the item, including information that affects its classification (e.g., 'plastic bottle with label' or 'banana peel')."
                    }, # <-- Added missing comma here
                    "category": {
                        "type": "string",
                        "enum": ["garbage", "recycling", "compost"]
                    }
                },
                "required": ["category", "object_description"],
                "additionalProperties": False
            }},
            max_tokens=50,
            temperature=0.0 
        )
        
        # Parse the JSON string returned by the API
        response_json = json.loads(response.choices[0].message.content)
        result = response_json.get("category", "unknown").lower()

        if result not in CATEGORIES:
            print(f"Unexpected category from AI: {result}. Defaulting to 'garbage'.")
            return "garbage"
            
        return result
        
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "garbage" # Fallback if API fails or network goes down

# ==========================================
# MAIN LOOP
# ==========================================

try:
    while True:
        print("\n--- Starting New Cycle ---")
        
        # STEP 1: Wait for significant change or button press
        print("Waiting for item...")
        pi.write(YELLOW_WAIT_LED, 1) # Turn on Wait LED
        
        ret, prev_frame = cap.read()
        if not ret:
            print("Failed to grab initial frame, retrying...")
            time.sleep(0.2)
            continue

        prev_frame = prev_frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        
        initial_guess = None
        current_frame = None
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                continue

            current_frame = current_frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Check for button press
            initial_guess = get_pressed_button()
            if initial_guess:
                print(f"Triggered by button press: {initial_guess}")
                break
                
            # Check for motion
            diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = cv2.countNonZero(thresh)
            
            if motion_score > MOTION_THRESHOLD:
                print("Triggered by motion detected!")
                print("Settling and clearing buffer...")
                settle_camera(wait_time=2.0)
                ret, current_frame = cap.read()
                if ret:
                    current_frame = current_frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                else:
                    print("Failed to grab settled frame — retrying motion loop")
                    prev_gray = gray
                    continue
                break
                
            prev_gray = gray
            time.sleep(0.1)

        # STEP 2: Send to AI and turn off Yellow LED
        if current_frame is None:
            print("No frame available for classification; restarting cycle.")
            pi.write(YELLOW_WAIT_LED, 0)
            continue

        actual_category = classify_image_with_ai(current_frame)
        print(f"AI classified item as: {actual_category}")
        pi.write(YELLOW_WAIT_LED, 0) 
        
        # STEP 3: Wait for button presses / Guessing Game
        guesses = []
        available_options = list(CATEGORIES.keys()) # Tracks which ones flash blue
        blue_led_state = 0
        last_flash_time = time.time()
        
        # If the user triggered the machine by pressing a button, that counts as their first guess
        if initial_guess:
            current_guess = initial_guess
            while get_pressed_button() == current_guess:
                time.sleep(0.05)
        else:
            current_guess = None

        while True:
            # --- Non-blocking Blue LED Flasher ---
            if current_guess is None: 
                if time.time() - last_flash_time > 0.5: # 500ms toggle
                    blue_led_state = 1 if blue_led_state == 0 else 0
                    for cat in available_options:
                        pi.write(CATEGORIES[cat]["blue_led"], blue_led_state)
                    last_flash_time = time.time()

            # Check for physical press
            if current_guess is None:
                current_guess = get_pressed_button()
                
            if current_guess:
                # Turn off all blue LEDs immediately to show a button was registered
                for cat in CATEGORIES:
                    pi.write(CATEGORIES[cat]["blue_led"], 0)

                if current_guess not in guesses:
                    guesses.append(current_guess)
                    
                if current_guess == actual_category:
                    print("User guessed correctly!")
                    pi.write(CATEGORIES[current_guess]["green_led"], 1)
                    break
                else:
                    print(f"Incorrect guess: {current_guess}. Flashing red.")
                    flash_red(current_guess)
                    # Remove from available options so it stops flashing blue
                    if current_guess in available_options:
                        available_options.remove(current_guess)
                    
                # Debounce: Wait for user to release the button
                while get_pressed_button() == current_guess:
                    time.sleep(0.05)
                
                # Reset states for the next loop
                current_guess = None
                blue_led_state = 0
                last_flash_time = time.time() 
                
            time.sleep(0.05)

        # Make sure blue LEDs are absolutely off after correct guess
        for cat in CATEGORIES:
            pi.write(CATEGORIES[cat]["blue_led"], 0)

        # STEP 4: Open the correct flap
        print(f"Opening {actual_category} flap...")
        servo_pin = CATEGORIES[actual_category]["servo"]
        pi.set_servo_pulsewidth(servo_pin, SERVO_OPEN)
        
        # STEP 5: Send guess results to API
        payload = {
            "category": actual_category,
            "guesses": guesses
        }
        print(f"Sending data to Reporting API: {payload}")
        try:
            requests.post(REPORT_API_URL, json=payload, timeout=5)
        except Exception as e:
            print(f"Reporting API Error: {e}")

        # STEP 6: Wait for some time, then close flap
        time.sleep(4) 
        print(f"Closing {actual_category} flap...")
        pi.set_servo_pulsewidth(servo_pin, SERVO_CLOSED)
        pi.write(CATEGORIES[actual_category]["green_led"], 0) 
        time.sleep(1) 

        # STEP 7: Repeat 

except KeyboardInterrupt:
    print("\nShutting down gracefully...")
finally:
    # Cleanup
    cap.release()
    pi.write(YELLOW_WAIT_LED, 0)
    for cat, pins in CATEGORIES.items():
        pi.write(pins["red_led"], 0)
        pi.write(pins["green_led"], 0)
        pi.write(pins["blue_led"], 0)
        pi.set_servo_pulsewidth(pins["servo"], 0)
    pi.stop()