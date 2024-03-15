"""I can't get the Udacity sim to work on Apple Silicon, so I'm writing a dummy requestor as a workaround."""
import base64
import socketio

sio = socketio.SimpleClient()
with open("./dataset/dataset/IMG/center_2019_04_02_19_25_33_671.jpg", "rb") as imageFile:
    str = base64.b64encode(imageFile.read())

sample_request = {
    "steering_angle": 0.0,
    "throttle": 0.0,
    "speed": 0.0,
    "image": str
}

sio = socketio.Client()
@sio.on("connect")
def on_connect():
    print("I'm connected!")
    sio.emit('telemetry', sample_request)

sio.connect('http://localhost:4567/')