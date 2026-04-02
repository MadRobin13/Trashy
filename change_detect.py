import cv2
import time

# Adjust this value based on your camera resolution and how sensitive 
# you want the motion detection to be.
MOTION_THRESHOLD = 10000 

ROI = (80, 38, 888, 901) # Set with the select_roi.py script (x, y, width, height)

def main():
    print("Initializing camera...")
    # 0 is usually the default built-in or USB webcam
    cap = cv2.VideoCapture(0)
    
    # Allow camera sensor to warm up
    time.sleep(2) 

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Camera ready. Watching for significant changes...")
    print("Press 'q' in any video window to quit.")

    # Capture the very first frame to establish a baseline
    ret, prev_frame = cap.read()


    if not ret:
        print("Failed to grab initial frame.")
        return

    # Crop to the region of interest (ROI) for more focused detection)
    prev_frame = prev_frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]

    # Convert to grayscale and blur to remove noise (prevents false positives)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    try:
        while True:
            ret, current_frame = cap.read()
            if not ret:
                continue
            
            # Crop to the same ROI
            current_frame = current_frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
            
            # Process the current frame
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Calculate the absolute difference between the previous and current frame
            diff = cv2.absdiff(prev_gray, gray)
            
            # Threshold the difference (pixels with > 25 difference become white, rest black)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Count the white pixels to get a "motion score"
            motion_score = cv2.countNonZero(thresh)
            
            # Display the visual feeds
            # Add a status text to the live feed
            display_frame = current_frame.copy()
            cv2.putText(display_frame, f"Motion Score: {motion_score}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # cv2.imshow("Live Feed", display_frame)
            # cv2.imshow("Motion Mask", thresh)
            
            # If motion exceeds the threshold, trigger the capture sequence
            if motion_score > MOTION_THRESHOLD:
                print(f"\nSignificant change detected! (Score: {motion_score})")
                print("Waiting 1 second for object to settle...")
                
                # --- 1 SECOND WAIT LOOP ---
                # We loop for 1 second reading frames so the live feed stays active
                # and the camera buffer doesn't give us a stale image.
                start_time = time.time()
                while time.time() - start_time < 2.0:
                    ret, wait_frame = cap.read()

                    if not ret: break

                    # Crop to the same ROI
                    wait_frame = wait_frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                    
                    # Show "WAITING" text on the live feed
                    cv2.putText(wait_frame, "MOTION DETECTED - WAITING...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # cv2.imshow("Live Feed", wait_frame)
                    cv2.waitKey(10) # Keep GUI responsive
                
                # --- CAPTURE THE SNAPSHOT ---
                ret, snapshot = cap.read()
                if ret:
                    # Crop to the same ROI
                    snapshot = snapshot[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                    # Display the exact frame we captured
                    cv2.imshow("Captured Snapshot", snapshot)
                    print(" ---> [Snapshot captured! Image would be sent to AI here]")
                
                # Cooldown so we don't immediately trigger again
                print("Cooldown for 5 seconds...")
                cv2.waitKey(5000) # Wait 5 seconds while keeping windows responsive
                
                # Reset the baseline to a fresh frame
                ret, fresh_frame = cap.read()
                if ret:
                    fresh_frame = fresh_frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                    prev_gray = cv2.cvtColor(fresh_frame, cv2.COLOR_BGR2GRAY)
                    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
                
                print("Resuming detection...")
                continue
                
            # Update the baseline frame for the next iteration
            prev_gray = gray
            
            # Check for 'q' key press to exit
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("\nQuit key pressed.")
                break

    except KeyboardInterrupt:
        print("\nStopping via KeyboardInterrupt...")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == "__main__":
    main()