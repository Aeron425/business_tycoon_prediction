from PIL import Image
import pytesseract
import pandas as pd
import os
import cv2

# Creating Data File
csv_file = "data.csv"
directory = "C:\\Users\\Shaunak\\Desktop\\VScode Python\\ORC\\Images"

data = {
    "Money": [],
    "Time": []
}
errors = 0

for image in os.listdir(directory):
    path = os.path.join(directory, image)

    # Getting Image
    img_array = cv2.imread(path)
    img_to_make_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, img_to_make_gray)
    print(path)

    img = Image.open(path)

    # Cropping Image to Sections
    price = img.crop((1300, 600, 1400, 675))
    time_crop = img.crop((1570, 500, 1650, 550))

    try:
        price_value = float(pytesseract.image_to_string(price).strip().lower())
        time_value = pytesseract.image_to_string(time_crop).strip().lower()
        print(price_value)
        print(time_value)

        hr = time_value[:2]
        mn = time_value[3:]
        time_str = hr + ':' + mn

        data["Money"].append(price_value)
        data["Time"].append(time_str)
    except ValueError:
        print(f"it's so over")
        errors += 1

df = pd.DataFrame(data)
df.to_csv(csv_file, index=False)
print(f"Total Errors: {errors}")

