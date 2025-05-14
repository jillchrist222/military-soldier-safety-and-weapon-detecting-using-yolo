import torch
import cv2
import time

# Loading the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to process the video and detect objects
def detect_objects_in_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects
        results = model(frame)
        
        # Convert results to pandas DataFrame
        df = results.pandas().xyxy[0]
        
        # Draw bounding boxes and labels on the frame
        for _, row in df.iterrows():
            x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']

            # Replace labels as per requirements
            if cls == 'person':
                cls = 'soldier'
            elif cls == 'cow':  # in my code, I have replaced 'cow' with 'soldier' because, it predicting some soldiers as cow
                cls = 'soldier'
            elif cls == 'truck':
                cls = 'tanker'

            label = f'{cls} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Writing the frame into the output file
        out.write(frame)
        
        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# input video path
video_path = 'military-clips.mp4'

# Call the function to detect objects in the video
detect_objects_in_video(video_path)
