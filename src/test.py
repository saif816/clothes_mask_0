import requests

# Base URL for the Flask application
base_url = "https://wziuufioe8vlz4-5000.proxy.runpod.net"

# Endpoint for the generate route
endpoint = "/generate"

# Full URL
url = f"{base_url}{endpoint}"

# Data to send in the POST request
data = {
    "person_image_url": "https://fitcheck-assets.s3.amazonaws.com/renderss/1test.png",
    "cloth_image_url": "https://fitcheck-assets.s3.amazonaws.com/renderss/images.jpeg",
    "cloth_type": "upper",  # Example type, replace with actual type
    "num_inference_steps": 50,  # Optional, default is 50
    "guidance_scale": 2.5,      # Optional, default is 2.5
    "seed": 42                 # Optional, default is 42
}

# Send POST request with no timeout
import time

start_time = time.time()

try:
    response = requests.post(url, json=data,timeout=30000)  # No timeout specified, waits indefinitely
    # Check if the request was successful
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken: {time_taken} seconds\n")
