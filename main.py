#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import paho.mqtt.client as mqtt

# Mqtt config
broker_ip = "192.168.50.210"
topic = "test/camera-crossline_detection"
client = mqtt.Client()
client.connect(broker_ip, 1883)
client.loop_start()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Camera config
slice_y = 300
slice_height = 80
slice_width = 640
threshold_value = 10000
previous_state = None
message_sent = False

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())

        roi = color_image[slice_y:slice_y + slice_height, 0:slice_width]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)

        left_side = binary[:, :slice_width // 2]
        right_side = binary[:, slice_width // 2:]
        left_white_pixels = np.sum(left_side == 255)
        right_white_pixels = np.sum(right_side == 255)

        if white_pixels > black_pixels:
            if left_white_pixels > threshold_value and right_white_pixels < threshold_value:
                current_state = "skrzyżowanie, skret w lewo"
            elif right_white_pixels > threshold_value and left_white_pixels < threshold_value:
                current_state = "skrzyżowanie, skret w prawo"
            elif left_white_pixels > threshold_value and right_white_pixels > threshold_value:
                current_state = "skrzyżowanie z dwóch stron"
        else:
            current_state = "brak skrzyżowania"

        if current_state != previous_state:
            if current_state != "brak skrzyżowania":
                client.publish(topic, current_state)
                print(f"Wysłano przez MQTT: {current_state}")
            previous_state = current_state

        cv2.rectangle(color_image, (0, slice_y), (slice_width, slice_y + slice_height), (0, 255, 0), 2)
        cv2.imshow('Original Image', color_image)
        cv2.imshow('Binary Image (ROI)', binary)

        if cv2.waitKey(1) == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()
