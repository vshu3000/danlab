#!/usr/bin/env python3
import sys
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

with open('query', 'r') as f:
    query = f.read().strip()

driver = webdriver.Chrome('../chromedriver')
driver.maximize_window()
driver.get("https://duckduckgo.com/?q=%s&t=h_&iar=images&iax=images&ia=images" % query)

time.sleep(3)
SCROLL_PAUSE_TIME = 0.5

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

for i in range(5):
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

time.sleep(3)


with open('urls', 'w') as f:
    for elem in driver.find_elements(By.CLASS_NAME, 'tile--img__img'):
        url = elem.get_attribute('data-src')
        f.write('http:')
        f.write(url)
        f.write('\n')
#elem.clear()
#elem.send_keys(Keys.RETURN)
#elem.send_keys("pycon")
#assert "No results found." not in driver.page_source
driver.close()
