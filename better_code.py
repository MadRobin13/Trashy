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
CATEGORIES = {
    "garbage":   {"red_led": 2,  "green_led": 3,  "servo": 4,  "button": 15},
    "recycling": {"red_led": 17, "green_led": 27, "servo": 22, "button": 18},
    "compost":   {"red_led": 10, "green_led": 9,  "servo": 11, "button": 23}
}

# Servo states (Pulse widths: 500-2500. 1500 is usually center)
SERVO_CLOSED = 1000
SERVO_OPEN = 2000

# Motion detection threshold
MOTION_THRESHOLD = 10000 

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

def classify_image_with_ai(frame):
    """Encodes the frame and sends it to OpenAI's Vision API."""
    print("Encoding image and sending to OpenAI...")
    
    # Convert OpenCV frame to JPEG, then to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # gpt-4o is fast, accurate, and has vision capabilities
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
                                "detail": "low" # 'low' uses fewer tokens and is faster, usually sufficient for object classification
                            }
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0 # Keep temperature low for highly deterministic output
        )
        
        # Extract and clean up the result
        result = response.choices[0].message.content.strip().lower()
        
        # Safety fallback in case the AI hallucinates outside of our 3 categories
        if result not in CATEGORIES:
            print(f"Unexpected AI response '{result}'. Defaulting to 'garbage'.")
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
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        
        initial_guess = None
        current_frame = None
        
        while True:
            ret, current_frame = cap.read()
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
                time.sleep(0.5) # Wait half a second for object to settle
                ret, current_frame = cap.read() # Grab a fresh, settled frame
                break
                
            prev_gray = gray
            time.sleep(0.1)

        # STEP 2: Send to AI and turn off Yellow LED
        actual_category = classify_image_with_ai(current_frame)
        print(f"AI classified item as: {actual_category}")
        pi.write(YELLOW_WAIT_LED, 0) 
        
        # STEP 3: Wait for button presses / Guessing Game
        guesses =[]
        
        # If the user triggered the machine by pressing a button, that counts as their first guess
        if initial_guess:
            current_guess = initial_guess
            while get_pressed_button() == current_guess:
                time.sleep(0.05)
        else:
            current_guess = None

        while True:
            if current_guess is None:
                current_guess = get_pressed_button()
                
            if current_guess:
                if current_guess not in guesses:
                    guesses.append(current_guess)
                    
                if current_guess == actual_category:
                    print("User guessed correctly!")
                    pi.write(CATEGORIES[current_guess]["green_led"], 1)
                    break
                else:
                    print(f"Incorrect guess: {current_guess}. Flashing red.")
                    flash_red(current_guess)
                    
                # Debounce: Wait for user to release the button
                while get_pressed_button() == current_guess:
                    time.sleep(0.05)
                current_guess = None
                
            time.sleep(0.05)

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
        pi.set_servo_pulsewidth(pins["servo"], 0)
    pi.stop()