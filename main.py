from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2

API_KEY = "lsyThMXXEDPYDWKzBHgg"
model  = "car-scratch-and-dent/3"
image  = "images/img.png"
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="lsyThMXXEDPYDWKzBHgg"
)
cfg = InferenceConfiguration(confidence_threshold=0.10)
result = CLIENT.infer(image, model_id="car-scratch-and-dent/3")

with CLIENT.use_configuration(cfg):
    res = CLIENT.infer(image, model_id=model)

img = cv2.imread(image)

for p in res.get("predictions", []):
    x, y, w, h = p["x"], p["y"], p["width"], p["height"]
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{p["class"]} {p["confidence"]:.2f}', (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", img)
cv2.resizeWindow("result", 900, 600)
cv2.waitKey(0)
cv2.destroyAllWindows()