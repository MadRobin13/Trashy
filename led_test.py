from gpiozero import RGBLED
import time

# Define pins for each category
CATEGORIES = {
    "garbage":   {"red_led": 3,  "green_led": 4,  "blue_led": 17,  "servo": 4,  "button": 19},
    "recycling": {"red_led": 22, "green_led": 10, "blue_led": 9,  "servo": 22, "button": 11},
    "compost":   {"red_led": 5, "green_led": 6,  "blue_led": 13, "servo": 11, "button": 15}
}

# Initialize RGBLED objects for each category
leds = {}
for category, pins in CATEGORIES.items():
    leds[category] = RGBLED(pins["red_led"], pins["green_led"], pins["blue_led"])

# Test function to set LEDs to specific colors
def test_led(category):
    
    # Cycle through red, green, and blue for the specified category
    if category in leds:
        print(f"Testing {category} LEDs...")
        leds[category].color = (1, 0, 0)  # Red
        time.sleep(1)
        leds[category].color = (0, 1, 0)  # Green
        time.sleep(1)
        leds[category].color = (0, 0, 1)  # Blue
        time.sleep(1)
        leds[category].off()  # Turn off after testing
    else:
        print(f"Category '{category}' not found in CATEGORIES.")

# Example usage
if __name__ == "__main__":
    test_led("garbage")
    time.sleep(1)
    test_led("recycling")
    time.sleep(1)
    test_led("compost")