import os
import time

import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- Setup Selenium ---
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # run in background
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# --- Open the catalog page ---
url = "https://www.keeneland.com/sales/2025/2/september-yearling-sale/catalog-table/"
driver.get(url)
time.sleep(5)  # wait for JS to render

# --- Prepare folder ---
os.makedirs("keeneland_images", exist_ok=True)

# --- Find the scrollable table container ---
table_container = driver.find_element(By.CSS_SELECTOR, "div.MuiTableContainer-root")

# Keep track of horses we've already processed
processed_hips = set()
scroll_pause = 1.0
last_height = driver.execute_script("return arguments[0].scrollHeight;", table_container)

while True:
    # Find all visible rows
    rows = table_container.find_elements(By.CSS_SELECTOR, "tr.MuiTableRow-root")

    for row in rows:
        try:
            # Get the HIP number for this row
            hip_elem = row.find_element(By.CSS_SELECTOR, "span#catalog-cell-horse-name-hip")
            hip_text = hip_elem.text.strip()
            if hip_text in processed_hips:
                continue  # already processed
            processed_hips.add(hip_text)

            # Scroll the row into view
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", row)
            time.sleep(0.3)

            # Find the Camera button in the last cell
            photo_buttons = row.find_elements(
                By.CSS_SELECTOR, "td:last-child button.btn-text svg[data-testid='CameraAltIcon']"
            )
            if not photo_buttons:
                print(f"No camera button for {hip_text}, skipping")
                continue

            button = photo_buttons[0].find_element(By.XPATH, "..")
            button.click()
            time.sleep(1)  # wait for modal

            # Grab the horse image from modal
            img = driver.find_element(By.CSS_SELECTOR, "img[alt^='Hip']")
            img_url = img.get_attribute("src")

            # Download image
            img_name = f"keeneland_images/{hip_text.replace(' ', '_')}.jpg"
            with open(img_name, "wb") as f:
                f.write(requests.get(img_url).content)
            print(f"Downloaded {img_name}")

            # Close modal
            close_button = driver.find_element(By.CSS_SELECTOR, "button.btn-close")

            close_button.click()
            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing {row.text[:50]}: {e}")
            continue

    # Scroll the container down
    driver.execute_script("arguments[0].scrollBy(0, 400);", table_container)
    time.sleep(scroll_pause)

    new_height = driver.execute_script("return arguments[0].scrollHeight;", table_container)
    if new_height == last_height:
        break  # reached bottom
    last_height = new_height

driver.quit()
print("Done!")
