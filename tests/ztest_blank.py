"""
Synchronous test on random data to test playwright
"""

import time

import numpy as np
from playwright.sync_api import sync_playwright

from pyeyes.viewers import ComparativeViewer as cv

TEST_PORT = 52890

data_dict = {
    "rng": np.random.randn(100, 100, 3),
    "zero": np.zeros((100, 100, 3)),
}

viewer = cv(data=data_dict, named_dims=list("xyz"))

# Start server in background thread
server = viewer.launch(
    show=False, start=True, threaded=True, verbose=True, port=TEST_PORT
)
time.sleep(0.5)

# Launch headless browser
pw = sync_playwright().start()
browser = pw.chromium.launch(headless=True)
page = browser.new_page()

# Test some interactions with the viewer
page.goto(f"http://localhost:{TEST_PORT}")
page.wait_for_load_state("networkidle")
print("Loaded chromium page.")
time.sleep(4)

# TODO: Add way to check each tab.
# print(f"Navigating to View tab.")
# ViewTab = page.get_by_role("tab", name="View")
# assert ViewTab.count() > 0, "View tab not found in page"
# ViewTab.click()

print("View tab clicked. Checking single View")
SingleViewCb = page.locator(".pyeyes-single-view input[type='checkbox']")
assert SingleViewCb.count() > 0, "Single View checkbox not found in page"
SingleViewCb.check()
print(f"Single View checkbox clicked. Display images: {viewer.slicer.display_images}")
time.sleep(1)

SingleViewCb.uncheck()
print(f"Single View checkbox unchecked. Display images: {viewer.slicer.display_images}")
time.sleep(3)

print("Stopping server.")
browser.close()
server.stop()
pw.stop()
print("Server stopped.")
