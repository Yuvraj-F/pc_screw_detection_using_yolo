"""
Running this script with no arguments opens 

This was done entirely by chat gpt with manual verification to ensure it works as expected. There can be issues where sometimes the augmented image
does not preserve the bounding box. 
"""

import cv2
import os
import sys
from config import *
import tkinter as tk
from tkinter import filedialog
from ultralytics.data.utils import visualize_image_annotations
from utils import hex_to_bgr 


IMAGE_DIR = TRAIN_DATA_DIR / "images"
LABEL_DIR = TRAIN_DATA_DIR / "labels"
VAL_IMAGE_DIR = VAL_DATA_DIR / "images"
VAL_LABEL_DIR = VAL_DATA_DIR / "labels"

OUTPUT_IMAGE_DIR = TRAIN_DATA_DIR / "images_cropped"
OUTPUT_LABEL_DIR = TRAIN_DATA_DIR / "labels_cropped"

MARGIN = 0

def load_labels(label_path):
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            labels.append((cls, x, y, w, h))
    return labels


def save_labels(label_path, labels):
    with open(label_path, "w") as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x} {y} {w} {h}\n")


def yolo_to_pixel(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return x1, y1, x2, y2


def pixel_to_yolo(x1, y1, x2, y2, crop_w, crop_h):
    x = ((x1 + x2) / 2) / crop_w
    y = ((y1 + y2) / 2) / crop_h
    w = (x2 - x1) / crop_w
    h = (y2 - y1) / crop_h
    return x, y, w, h

def get_safe_crop(x1, y1, x2, y2, img_w, img_h, margin=MARGIN):

    bw = x2 - x1
    bh = y2 - y1

    # expand around box
    x1 -= bw * margin
    x2 += bw * margin 
    y1 -= bh * margin
    y2 += bh * margin

    # clamp
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(img_w, x2))
    y2 = int(min(img_h, y2))

    return x1, y1, x2, y2

def select_file():
    root = tk.Tk()
    root.withdraw()  # hide main window

    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return None, None

    # extract filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]

    return file_path, filename

def view_selected_image(img_path):
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".JPG", ".txt").replace(".png", ".txt")   

    # From ultralytics
    visualize_image_annotations(
        img_path,  # Input image path.
        label_path,  # Annotation file path for the image.
        CLASS_NAMES,
    )

def browse_dataset():
    while True:
        option = input("Which dataset would you like to browse? Training[1] | Validation[2]: ")
        if option == "1":
            img_dir = IMAGE_DIR
            label_dir = LABEL_DIR
            break
        elif option == "2":
            img_dir = VAL_IMAGE_DIR
            label_dir = VAL_LABEL_DIR
            break
        else:
            print("Inavlid option")

    print("\n---Controls---\nNext: n | Previous: p | Quit: q")

    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    idx = 0
    while True:

        img_file = image_files[idx]
        filename = os.path.splitext(img_file)[0]

        img_path = img_dir / img_file
        label_path = label_dir / (filename + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed: {img_path}")
            continue

        h, w = img.shape[:2]

        # draw annotations if they exist
        if label_path.exists():
            labels = load_labels(label_path)

            for cls, x, y, bw, bh in labels:

                x1, y1, x2, y2 = yolo_to_pixel(x, y, bw, bh, w, h)
                color = tuple(int(c*0.5) for c in hex_to_bgr(CLASS_COLORS.get(cls, None)))

                cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    5
                )

                text = CLASS_NAMES.get(cls, str(cls))
                cv2.putText(
                    img,
                    text,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    color,
                    4
                )

        display = cv2.resize(img, (900, 900))
        cv2.imshow("Dataset Browser", display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = min(idx + 1, len(image_files) - 1)
        elif key == ord('p'):
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()

def crop():
    _, filename = select_file()
    img_path = IMAGE_DIR / (filename + ".jpg")
    label_path = LABEL_DIR / (filename + ".txt")
  
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    labels = load_labels(label_path)

    for i, (cls, x, y, bw, bh) in enumerate(labels):

        # convert YOLO → pixel
        x1, y1, x2, y2 = yolo_to_pixel(x, y, bw, bh, w, h)

        # create crop that GUARANTEES full box inclusion
        cx1, cy1, cx2, cy2 = get_safe_crop(x1, y1, x2, y2, w, h)

        cropped = img[cy1:cy2, cx1:cx2]
        ch, cw = cropped.shape[:2]

        new_labels = []

        # adjust ALL boxes that fall inside crop
        for cls2, x2n, y2n, w2n, h2n in labels:
            bx1, by1, bx2, by2 = yolo_to_pixel(x2n, y2n, w2n, h2n, w, h)

            # check intersection
            ix1 = max(bx1, cx1)
            iy1 = max(by1, cy1)
            ix2 = min(bx2, cx2)
            iy2 = min(by2, cy2)

            if ix1 >= ix2 or iy1 >= iy2:
                continue

            # shift into cropped frame
            ix1 -= cx1
            ix2 -= cx1
            iy1 -= cy1
            iy2 -= cy1

            ny, nx, nw, nh = pixel_to_yolo(ix1, iy1, ix2, iy2, cw, ch)
            new_labels.append((cls2, ny, nx, nw, nh))

            cv2.rectangle(cropped,
                          (int(ix1), int(iy1)),
                          (int(ix2), int(iy2)),
                          (0, 255, 0), 2)

        cv2.imshow(f"Crop around box {i}", cv2.resize(cropped, (800, 800)))
        cv2.waitKey(0)
 
    cv2.destroyAllWindows()

def auto_crop_old():

    OUTPUT_IMAGE_DIR.mkdir(exist_ok=True)
    OUTPUT_LABEL_DIR.mkdir(exist_ok=True)

    for img_file in os.listdir(IMAGE_DIR):

        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filename = os.path.splitext(img_file)[0]

        img_path = IMAGE_DIR / img_file
        label_path = LABEL_DIR / (filename + ".txt")

        # skip if label file doesn't exist
        if not label_path.exists():
            print(f"Skipping (no label): {filename}")
            continue

        print(f"Processing: {filename}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load: {img_path}")
            continue

        h, w = img.shape[:2]
        labels = load_labels(label_path)

        for i, (cls, x, y, bw, bh) in enumerate(labels):

            # convert YOLO → pixel
            x1, y1, x2, y2 = yolo_to_pixel(x, y, bw, bh, w, h)

            # create crop that GUARANTEES full box inclusion
            cx1, cy1, cx2, cy2 = get_safe_crop(x1, y1, x2, y2, w, h)

            cropped = img[cy1:cy2, cx1:cx2]
            ch, cw = cropped.shape[:2]

            new_labels = []

            for cls2, x2n, y2n, w2n, h2n in labels:
                bx1, by1, bx2, by2 = yolo_to_pixel(x2n, y2n, w2n, h2n, w, h)

                # intersection
                ix1 = max(bx1, cx1)
                iy1 = max(by1, cy1)
                ix2 = min(bx2, cx2)
                iy2 = min(by2, cy2)

                if ix1 >= ix2 or iy1 >= iy2:
                    continue

                # shift into crop space
                ix1 -= cx1
                ix2 -= cx1
                iy1 -= cy1
                iy2 -= cy1

                ny, nx, nw, nh = pixel_to_yolo(ix1, iy1, ix2, iy2, cw, ch)
                new_labels.append((cls2, ny, nx, nw, nh))

            # skip empty crops (shouldn't happen, but safe)
            if len(new_labels) == 0:
                continue

            # new filename
            new_name = f"{filename}_{i+1}"

            new_img_path = OUTPUT_IMAGE_DIR / (new_name + ".jpg")
            new_label_path = OUTPUT_LABEL_DIR / (new_name + ".txt")

            # save
            cv2.imwrite(str(new_img_path), cropped)
            save_labels(new_label_path, new_labels)

    print("Done.")

def auto_crop_all():

    OUTPUT_IMAGE_DIR.mkdir(exist_ok=True)
    OUTPUT_LABEL_DIR.mkdir(exist_ok=True)

    MARGINS = [1, 2, 3, 4, 5]

    for img_file in os.listdir(IMAGE_DIR):

        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filename = os.path.splitext(img_file)[0]

        img_path = IMAGE_DIR / img_file
        label_path = LABEL_DIR / (filename + ".txt")

        # skip if label file doesn't exist
        if not label_path.exists():
            print(f"Skipping (no label): {filename}")
            continue

        print(f"Processing: {filename}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load: {img_path}")
            continue

        h, w = img.shape[:2]
        labels = load_labels(label_path)

        save_count = 0
        for margin_scale in MARGINS:
            for i, (cls, x, y, bw, bh) in enumerate(labels):

                # convert YOLO → pixel
                x1, y1, x2, y2 = yolo_to_pixel(x, y, bw, bh, w, h)

                # create crop that GUARANTEES full box inclusion
                cx1, cy1, cx2, cy2 = get_safe_crop(x1, y1, x2, y2, w, h, margin=margin_scale)

                cropped = img[cy1:cy2, cx1:cx2]
                ch, cw = cropped.shape[:2]

                new_labels = []

                for cls2, x2n, y2n, w2n, h2n in labels:
                    bx1, by1, bx2, by2 = yolo_to_pixel(x2n, y2n, w2n, h2n, w, h)

                    # intersection
                    ix1 = max(bx1, cx1)
                    iy1 = max(by1, cy1)
                    ix2 = min(bx2, cx2)
                    iy2 = min(by2, cy2)

                    if ix1 >= ix2 or iy1 >= iy2:
                        continue

                    # shift into crop space
                    ix1 -= cx1
                    ix2 -= cx1
                    iy1 -= cy1
                    iy2 -= cy1

                    ny, nx, nw, nh = pixel_to_yolo(ix1, iy1, ix2, iy2, cw, ch)
                    new_labels.append((cls2, ny, nx, nw, nh))

                # skip empty crops (shouldn't happen, but safe)
                if len(new_labels) == 0:
                    continue

                save_count += 1
                # new filename
                new_name = f"{filename}_m{margin_scale}_{save_count}"

                new_img_path = OUTPUT_IMAGE_DIR / (new_name + ".jpg")
                new_label_path = OUTPUT_LABEL_DIR / (new_name + ".txt")

                # save
                cv2.imwrite(str(new_img_path), cropped)
                save_labels(new_label_path, new_labels)

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print(f"This is a utility for viewing annotated YOLO training images. It expects the images and annotations to be in the following directories:\n\nTraining Data: {IMAGE_DIR} and {LABEL_DIR}\nVaidation Data: {VAL_IMAGE_DIR} {VAL_LABEL_DIR}\n")
        print("Usage: python crop.py [Option]\n")
        print("Options:")
        print("browse   View annotated images. Useful for verifying results from crop or auto_crop.")
        print("view     Opens the selected image with annotations loaded from labels directory.")
        print("crop     Crop images while preserving annotations. Results are saved in a new directory.")
        print("auto_crop    Crops the entire dataset automatically. Results are saved in a new directory.")
        
        sys.exit()

    if mode == "view":
        view_selected_image(select_file()[0])
    elif mode == "auto_crop":
        auto_crop_all()
    elif mode == "crop":
        crop()
    elif mode == "browse":
        browse_dataset()

