import cv2
import numpy as np
import argparse

# This function is just a placeholder for the trackbar.
def nothing(x):
    pass

# --- Main Tuning Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canny Edge Detection Tuner")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(parser.parse_args())

    image = cv2.imread(args["image"])
    if image is None:
        print(f"Error: Could not load image at {args['image']}")
        exit()

    # Resize for consistent display
    ratio = image.shape[0] / 800.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 800))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- Interactive Tuning UI Setup ---
    cv2.namedWindow("Canny Edge Tuner")
    cv2.createTrackbar("Threshold 1", "Canny Edge Tuner", 75, 255, nothing)
    cv2.createTrackbar("Threshold 2", "Canny Edge Tuner", 200, 255, nothing)

    print("--> Adjust the sliders to get a clean outline of the sheet.")
    print("--> Press 'q' to quit once you are done.")

    while True:
        # Get current positions of the two trackbars
        thresh1 = cv2.getTrackbarPos("Threshold 1", "Canny Edge Tuner")
        thresh2 = cv2.getTrackbarPos("Threshold 2", "Canny Edge Tuner")

        # Apply Canny edge detection with the current slider values
        edged = cv2.Canny(blurred, thresh1, thresh2)
        
        # --- Find and draw contours based on the live-tuned edges ---
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        screen_contour = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.contourArea(c) > (image.shape[0] * image.shape[1] * 0.1):
                screen_contour = approx
                break
        
        # Create a copy of the original image to draw on
        output_image = image.copy()
        if screen_contour is not None:
            cv2.drawContours(output_image, [screen_contour], -1, (0, 255, 0), 3)

        # Display the results
        cv2.imshow("Original with Contour", output_image)
        cv2.imshow("Canny Edges", edged)

        # Wait for a key press, if 'q', break the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"\nFinal values -> Threshold 1: {thresh1}, Threshold 2: {thresh2}")