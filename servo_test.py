from gpiozero import Servo
import time

# Define pins for each category
CATEGORIES = {
    "garbage":   {"red_led": 3,  "green_led": 4,  "blue_led": 17,  "servo": 17,  "button": 19},
    "recycling": {"red_led": 22, "green_led": 10, "blue_led": 9,  "servo": 18, "button": 11},
    "compost":   {"red_led": 5, "green_led": 6,  "blue_led": 13, "servo": 23, "button": 15}
}

# Initialize Servo objects for each category
servos = {}
for category, pins in CATEGORIES.items():
    servos[category] = Servo(pins["servo"])

# Test function to move servos to specific positions
def test_servo(category):
    # Move the servo to different positions for the specified category
    if category in servos:
        print(f"Testing {category} servo...")
        servos[category].min()  # Move to minimum position
        time.sleep(1)
        servos[category].mid()  # Move to middle position
        time.sleep(1)
        servos[category].min()  # Move back to minimum position
        servos[category].detach()  # Detach after testing
    else:
        print(f"Category '{category}' not found in CATEGORIES.")

# Example usage
if __name__ == "__main__":
    test_servo("garbage")
    time.sleep(1)
    test_servo("recycling")
    time.sleep(1)
    test_servo("compost")
    time.sleep(1)