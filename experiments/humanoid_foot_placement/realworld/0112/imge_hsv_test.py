import cv2
import os 
import numpy as np


def detect_foot_targets(rgb_image):

    # Convert to grayscale and denoise
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, (90, 88, 100), (120, 255, 255))

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours for visualization
    contours = contours[0] if len(contours) == 2 else contours[1]
    targets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Minimum area threshold to filter noise
            continue

        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)

        if MA / ma < 0.5:  # Filter out non-elliptical shapes
            continue

        targets.append((x, y, MA, ma, angle))

    targets = np.array(targets, dtype=np.float32)

    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    for target in targets:
        x, y, MA, ma, angle = target
        bgr = cv2.ellipse(bgr, (int(x), int(y)), (int(MA / 2), int(ma / 2)), angle, 0, 360, (0, 0, 255), 2)
    return bgr, cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)


def main():
    # get all jpg files in the directory
    image_dir = os.path.dirname(os.path.abspath(__file__)) + "/images"

    # sort image files by name ending number
    image_list = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_list.append(file_name)
    image_list = sorted(image_list, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    # make a gif from all images
    gif_frames = []

    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bgr_image, mask = detect_foot_targets(rgb_image)
        bgr_image = cv2.putText(bgr_image, f"Image: {image_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        gif_frames.append(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))

        cv2.imshow("Original Image", bgr_image)
        cv2.imshow("HSV Image", mask)
        cv2.waitKey(0)
    # save gif
    import imageio
    gif_path = os.path.join(image_dir, "foot_target_detection.gif")
    imageio.mimsave(gif_path, gif_frames[:300:3], fps=10)


if __name__ == "__main__":
    main()
