import cv2
import numpy as np
import argparse

def order_points(pts):
    """
    Orders the four corner points in a consistent order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, corners):
    """
    Applies a perspective transform to an image to get a bird's-eye view.
    """
    ordered_corners = order_points(corners)
    (tl, tr, br, bl) = ordered_corners
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMR Sheet Perspective Correction")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(parser.parse_args())

    image = cv2.imread(args["image"])
    ratio = image.shape[0] / 500.0
    original_image = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # --- IMPROVEMENT 1: Use a slightly larger blur kernel ---
    # This helps remove more high-frequency noise like wood grain.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 75, 200)

    print("STEP 1: Edge Detection")
    # --- IMPROVEMENT 2: Add a debug window for the edge map ---
    # This shows you what the contour finder is "seeing".
    cv2.imshow("Edged", edged)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] # Get more candidates
    
    screen_contour = None

    # --- IMPROVEMENT 3: Loop through more contours and check area ---
    # We are looking for the largest quadrilateral in the image.
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        
        # The contour for the sheet should have 4 points.
        if len(approx) == 4:
            # Check if this contour is reasonably large
            if cv2.contourArea(c) > 5000: # This threshold prevents small objects
                screen_contour = approx
                break # We found it, so we can stop looping
            
    if screen_contour is None:
        print("Could not find a 4-point contour. Please check image quality or lighting.")
        exit()

    print("STEP 2: Found contour of the sheet")
    cv2.drawContours(image, [screen_contour], -1, (0, 255, 0), 2)
    cv2.imshow("Sheet Contour", image) # Show what contour was selected
    
    rectified_image = perspective_transform(original_image, screen_contour.reshape(4, 2) * ratio)
    
    print("STEP 3: Applied perspective transform")

    output_filename = "rectified_image.png"
    cv2.imwrite(output_filename, rectified_image)
    print(f"Successfully processed image. Rectified sheet saved as '{output_filename}'")
    
    cv2.imshow("Original Image", cv2.resize(original_image, (600, 800)))
    cv2.imshow("Rectified Image", cv2.resize(rectified_image, (600, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()