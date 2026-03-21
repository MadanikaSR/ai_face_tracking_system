import cv2
import json
import os
import sys
from core.pipeline import Pipeline
from utils.helpers import Helpers
from ui.launcher import LauncherUI

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path="config.json"):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def get_input_source(config):
    """Determines the cv2.VideoCapture source based on config."""
    input_type = config.get("input_type", "webcam")
    
    if input_type == "video":
        source = config.get("video_path", "")
        if not os.path.exists(source):
            print(f"Error: Video file not found at {source}")
            return None, input_type, source
        return source, input_type, source
        
    elif input_type == "rtsp":
        source = config.get("rtsp_url", "")
        if not source:
            print("Error: RTSP URL not provided in config.")
            return None, input_type, source
        return source, input_type, source
        
    else: # Default to webcam
        source = config.get("webcam_index", 0)
        return source, input_type, f"Webcam {source}"

def main():
    # 1. Load current config
    current_config = load_config()
    
    # 2. Start UI Launcher
    # We pass a temporary pipeline to the launcher just for the reset button
    temp_pipeline = Pipeline(current_config)
    launcher = LauncherUI(current_config, temp_pipeline.reset_system)
    selected_config = launcher.run()
    
    if not selected_config:
        print("Launcher closed. Exiting.")
        sys.exit(0)
        
    # 3. Save selected config for next time
    save_config(selected_config)
    
    # 4. Initialize Pipeline with selected config
    pipeline = Pipeline(selected_config)
    
    # 5. Handle Reset on Start if checked
    if selected_config.get("reset_on_start", False):
        pipeline.reset_system()
        print("System reset performed on startup.")

    # 6. Setup Input Source
    source, mode, description = get_input_source(selected_config)
    if source is None:
        sys.exit(1)
        
    print(f"Starting system in {mode} mode...")
    print(f"Source: {description}")
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open input source {description}")
        sys.exit(1)

    print("Stream started successfully. Press 'q' to quit.")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break
            
            frame_count += 1
            
            # Process frame through pipeline
            tracked_objects, recognitions, confidences, raw_detections = pipeline.process_frame(frame, frame_count)
            
            # Draw detections (includes RAW debug layer)
            Helpers.draw_detections(frame, tracked_objects, recognitions, confidences, raw_detections)
            
            # Show output
            cv2.imshow("AI Face Tracking System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Unexpected Runtime Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("System shut down.")

if __name__ == "__main__":
    main()
