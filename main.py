from PIL import Image
import pytesseract
import pandas as pd
import os
import cv2
import numpy as np


def pytesseract_preprocessing(cropped_image, padding=10):
    img = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(img)


def loop(directory, csv_file):


    errored_imgs = []
    errors = 0

    for image in sorted(os.listdir(directory)):
        path = os.path.join(directory, image)
        img = Image.open(path).convert("RGB")
        price_crop = img.crop((1300, 590, 1450, 685))
        time_crop = img.crop((1570, 500, 1650, 550))

        price_crop.save(f"resources/cropped_images/price_{image}")
        time_crop.save(f"resources/cropped_images/time_{image}")

        try:
            price = pytesseract_preprocessing(price_crop)
            time = pytesseract_preprocessing(time_crop)

            price.save(f"resources/processed_images/price_{image}")
            time.save(f"resources/processed_images/time_{image}")

            price_raw = pytesseract.image_to_string(
                price, config=r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.'
            ).strip()
            price_value = float(price_raw)

            time_value = pytesseract.image_to_string(
                time, config=r'--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:'
            ).strip()
            time_str = time_value[:2] + ':' + time_value[3:]

            row = pd.DataFrame({"Money": [price_value], "Time": [time_str]})
            row.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

            print(price_value)
            print(time_str)

        except Exception as e:
            errored_imgs.append(image)
            print(f"Error processing {image}: {e}")
            errors += 1

    print(f"Total Errors: {errors}")
    print(f"Errored images: {errored_imgs}")


def main():
    csv_file = "data.csv"
    directory = "resources/images"
    loop(directory, csv_file)


main()