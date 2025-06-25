import cv2 # OpenCV library (for drawing and displaying)
import numpy as np # Numerical Python library
from ultralytics import YOLO # YOLOv8 library
import av # PyAV library for FFmpeg integration
import av.error # Import the error module for specific exceptions

# --- Configuration ---
model = YOLO('yolov8n.pt') # Load the pre-trained YOLOv8 model
PERSON_CLASS_ID = 0        # Class ID for 'person'
CONF_THRESHOLD = 0.27      # Confidence threshold

# --- RTSP Camera Configuration ---/
# IMPORTANT: Replace this with your actual RTSP link, including password if needed!
# Remember: rtsp://username:password@ip_address:port/path_to_stream
RTSP_URL = 'enter your rtsp'
# Example with actual password:
# RTSP_URL = 'rtsp://admin:MySecurePass123@RTSP-ATPL-908604-AIPTZ.torqueverse.dev:8604/ch0_0.264'

# --- Main Logic ---

def run_person_counter_pyav():
    print(f"Attempting to open RTSP stream from {RTSP_URL} using PyAV...")
    container = None
    try:
        # Open the RTSP stream
        # timeout: set a timeout for connecting and reading (in seconds)
        # rtsp_transport='tcp': Often more reliable than 'udp' for problematic streams.
        # This argument requires PyAV 8.0+
        container = av.open(RTSP_URL, options={'rtsp_transport': 'tcp', 'stimeout': '10000000'})
        print("RTSP stream opened successfully with PyAV. Press 'q' to quit.")

        # Get the first video stream
        video_stream = next(s for s in container.streams if s.type == 'video')

        while True:
            # Read a frame packet from the stream
            # Iterate through demuxed packets, then decode frames from packets
            for packet in container.demux(video_stream):
                for frame_pyav in packet.decode():
                    # Convert PyAV frame to OpenCV (NumPy) format
                    # frame_pyav.to_ndarray(format='bgr24') converts to BGR NumPy array
                    frame = frame_pyav.to_ndarray(format='bgr24')
                    frame = cv2.resize(frame, (960, 600))  # Resize to 1080x960


                    # --- Perform Object Detection with YOLOv8 ---
                    results = model(frame, conf=CONF_THRESHOLD, classes=[PERSON_CLASS_ID])
                    #results = model(frame, conf=CONF_THRESHOLD)

                    people_on_screen = 0
                    if results and results[0].boxes:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])

                            if class_id == PERSON_CLASS_ID:
                                people_on_screen += 1
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f'Person: {confidence:.2f}'
                                cv2.putText(frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # label = f'{model.names[class_id]}: {confidence:.2f}'
                            # cv2.putText(frame, label, (x1, y1 - 10),qq
                            #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


                    # Display the current count of people on the frame
                    count_text = f'People on Screen: {people_on_screen}'
                    cv2.putText(frame, count_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Display the frame
                    cv2.imshow('Live People Counter (PyAV RTSP)', frame)

                    # Wait for a key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Exiting loop.")
                        return # Exit the function and cleanup

                # Small break to prevent tight loop if no frames are decoded from a packet
                # (though the demux/decode structure usually handles this)
                # cv2.waitKey(1) already serves this purpose somewhat.

    # Catch specific PyAV errors
    except av.error.FFmpegError as e:
        print(f"FFmpeg error with PyAV: {e}")
        print("This often means the RTSP stream is inaccessible, invalid, or requires specific codec/protocol settings.")
        print("Ensure FFmpeg is correctly installed on your system and its bin directory is in PATH.")
        print("Double-check the RTSP URL, especially credentials and stream path, and test it thoroughly in VLC.")
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
    finally:
        if container:
            container.close() # Close the stream cleanly
        cv2.destroyAllWindows()
        print("Exiting Person Counter.")

# Run the function
if __name__ == '__main__':
    run_person_counter_pyav()