import cv2
import numpy as np
import argparse

def order_points(pts):
    """Orders the four corner points of the bubble grid."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def perspective_transform(image, corners):
    """Applies a perspective transform to the image."""
    ordered_corners = order_points(corners)
    (tl, tr, br, bl) = ordered_corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMR Sheet detection using Hough Circles")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = vars(parser.parse_args())

    original_image = cv2.imread(args["image"])
    if original_image is None:
        print(f"Error: Could not load image at {args['image']}")
        exit()

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    print("STEP 1: Detecting circles...")
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=25, minRadius=5, maxRadius=15)

    if circles is None:
        print("Could not detect any circles. Try adjusting HoughCircles parameters.")
        exit()

    print(f"STEP 2: Found {len(circles[0])} circles. Calculating precise corners...")
    
    points = circles[0][:, :2]

    rect = cv2.minAreaRect(points.astype(np.int32))
    
    center, (width, height), angle = rect
    
    if width < height:
        angle += 90
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_points = cv2.transform(np.array([points]), M)[0]
    
    min_x, min_y = np.min(rotated_points, axis=0)
    max_x, max_y = np.max(rotated_points, axis=0)
    
    # --- FIX: Add padding to account for bubble radius ---
    # This expands the box to ensure the entire bubble is included.
    padding = 20 # pixels
    
    upright_corners = np.array([
        [min_x - padding, min_y - padding], 
        [max_x + padding, min_y - padding],
        [max_x + padding, max_y + padding], 
        [min_x - padding, max_y + padding]
    ], dtype="float32")
    
    inv_M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    sheet_corners = cv2.transform(np.array([upright_corners]), inv_M)[0]

    # Draw debug information
    debug_image = original_image.copy()
    cv2.drawContours(debug_image, [sheet_corners.astype(int)], -1, (0, 0, 255), 3)
    cv2.imshow("Precise Corners", cv2.resize(debug_image, (600, 800)))

    # Apply the final perspective transform
    rectified_image = perspective_transform(original_image, sheet_corners)
    print("STEP 3: Applied final perspective transform.")

    output_filename = "rectified_image_final.png"
    cv2.imwrite(output_filename, rectified_image)
    print(f"Successfully processed image. Final rectified sheet saved as '{output_filename}'")
    
    cv2.imshow("Final Rectified Image", cv2.resize(rectified_image, (600, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()