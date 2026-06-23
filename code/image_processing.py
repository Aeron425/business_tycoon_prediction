from PIL import Image
import pytesseract
import pandas as pd
import os
import cv2
import numpy as np
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def pytesseract_preprocessing(cropped_image, padding=10):
    img = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(img)


def filename_to_datetime(filename):
    parts = filename.replace(".jpeg", "").replace(".jpg", "").split()
    
    if len(parts) not in (6, 7):
        return datetime.min

    if len(parts) == 7:
        m, d, y, h, minute, s, ampm = parts
        hour = int(h) + 12 if ampm.upper() == "PM" and int(h) < 12 else int(h)
        return datetime(int(y), int(m), int(d), hour, int(minute), int(s))
        
    else:
        if len(parts[0]) == 4:
            y, m, d, h, minute, s = parts
        else:
            m, d, y, h, minute, s = parts
            
        return datetime(int(y), int(m), int(d), int(h), int(minute), int(s))




def loop(directory, csv_file):
    os.makedirs("resources/cropped_images", exist_ok=True)
    os.makedirs("resources/processed_images", exist_ok=True)
    errored_imgs = []
    errors = 0

    for image in sorted(os.listdir(directory), key=filename_to_datetime):
        time = filename_to_datetime(image)
        path = os.path.join(directory, image)
        img = Image.open(path).convert("RGB")
        price_crop = img.crop((1300, 590, 1450, 685))

        price_crop.save(f"resources/cropped_images/price_{image}")

        try:
            price = pytesseract_preprocessing(price_crop)
            price.save(f"resources/processed_images/price_{image}")

            price_raw = pytesseract.image_to_string(price, config=r"--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.").strip()
            price_value = float(price_raw)

            row = pd.DataFrame({"Money": [price_value], "Time": [time]})
            row.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)

            print(price_value)
            print(time)

        except Exception as e:
            errored_imgs.append(image)
            print(f"Error processing {image}: {e}")
            errors += 1

    print(f"Total Errors: {errors}")
    print(f"Errored images: {errored_imgs}")


def main():
    csv_file = "resources/data/data.csv"
    directory = "resources/images"
    loop(directory, csv_file)


main()