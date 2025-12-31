import cv2
import mediapipe as mp
import numpy as np
import pyautogui

screen_width, screen_height = pyautogui.size()
print(screen_width, screen_height)

# initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# mouth landmark indices
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
UPPER_LIP = 0
LOWER_LIP = 17
FOREHEAD_TOP = 10
CHIN_BOTTOM = 152

# calibration states
WAIT_FOR_START = 0
CALIBRATE_NEUTRAL = 1
WAIT_FOR_SMILE = 2
CALIBRATE_SMILE = 3
CALIBRATED = 4

# calibration data
calibration_state = WAIT_FOR_START
calibration_data = {
    'neutral_width': [],
    'neutral_lift': [],
    'smile_width': [],
    'smile_lift': [],
    'min_width': None,
    'max_width': None,
    'min_lift': None,
    'max_lift': None
}

def calculate_distance(point1, point2):
    """calculate euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_raw_metrics(landmarks, image_width, image_height):
    """
    calculate raw mouth metrics without scoring
    returns: mouth_width_to_face_height, corner_lift_to_face_height
    """
    # get key mouth points
    mouth_left = landmarks[MOUTH_LEFT]
    mouth_right = landmarks[MOUTH_RIGHT]
    mouth_top = landmarks[MOUTH_TOP]
    mouth_bottom = landmarks[MOUTH_BOTTOM]
    
    # get forehead and chin for normalization
    forehead = landmarks[FOREHEAD_TOP]
    chin = landmarks[CHIN_BOTTOM]
    forehead_y = forehead.y * image_height
    chin_y = chin.y * image_height
    face_height = abs(chin_y - forehead_y)
    
    # convert normalized coordinates to pixel coordinates
    left_x = mouth_left.x * image_width
    left_y = mouth_left.y * image_height
    right_x = mouth_right.x * image_width
    right_y = mouth_right.y * image_height
    top_y = mouth_top.y * image_height
    bottom_y = mouth_bottom.y * image_height

    # calculate mouth width
    mouth_width = calculate_distance([left_x, left_y], [right_x, right_y])
    
    # calculate corner lift
    center_y = (top_y + bottom_y) / 2
    avg_corner_y = (left_y + right_y) / 2
    corner_lift = center_y - avg_corner_y
    
    # normalize to face height
    mouth_width_to_face_height = mouth_width / face_height
    corner_lift_to_face_height = corner_lift / face_height
    
    return mouth_width_to_face_height, corner_lift_to_face_height

def linear_map(value, min_val, max_val):
    """map value from [min_val, max_val] to [0, 100]"""
    if max_val == min_val:
        return 50  # avoid division by zero
    return (value - min_val) / (max_val - min_val) * 100

def calculate_smile_score(landmarks, image_width, image_height, debug=False):
    """
    calculate smile score from 0-100 based on calibration
    """
    width_metric, lift_metric = calculate_raw_metrics(landmarks, image_width, image_height)
    
    # if not calibrated, return raw value
    if calibration_state != CALIBRATED:
        return 0, {'width_metric': width_metric, 'lift_metric': lift_metric}
    
    # map to 0-100 using calibration
    width_score = linear_map(width_metric, 
                             calibration_data['min_width'], 
                             calibration_data['max_width'])
    lift_score = linear_map(lift_metric, 
                            calibration_data['min_lift'], 
                            calibration_data['max_lift'])
    
    # average the scores
    smile_score = (width_score + lift_score) / 2
    
    if debug:
        return int(smile_score), {
            'width_metric': width_metric,
            'lift_metric': lift_metric,
            'width_score': width_score,
            'lift_score': lift_score,
            'min_width': calibration_data['min_width'],
            'max_width': calibration_data['max_width'],
            'min_lift': calibration_data['min_lift'],
            'max_lift': calibration_data['max_lift']
        }
    
    return int(smile_score), {}

def finalize_calibration():
    """calculate min/max from calibration samples"""
    if len(calibration_data['neutral_width']) > 0 and len(calibration_data['smile_width']) > 0:
        all_widths = calibration_data['neutral_width'] + calibration_data['smile_width']
        all_lifts = calibration_data['neutral_lift'] + calibration_data['smile_lift']
        
        calibration_data['min_width'] = min(all_widths)
        calibration_data['max_width'] = max(all_widths)
        calibration_data['min_lift'] = min(all_lifts)
        calibration_data['max_lift'] = max(all_lifts)
        
        print(f"\ncalibration complete! ^^")
        print(f"width range: {calibration_data['min_width']:.2f} to {calibration_data['max_width']:.2f}")
        print(f"lift range: {calibration_data['min_lift']:.4f} to {calibration_data['max_lift']:.4f}\n")
        return True
    return False

# start webcam
cap = cv2.VideoCapture(0)

# create named window
window_name = 'smile detector with calibration ^^'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# the code for centering window doesn't really work but whatever, it's okay
window_width = 1280
window_height = 720

# calculate center position
x_pos = (screen_width - window_width) // 2
y_pos = (screen_height - window_height) // 2

# move window to center
cv2.moveWindow(window_name, x_pos, y_pos)

# debug mode toggle
DEBUG_MODE = True

print("=== smile detector with calibration ===")
print("step 1: press 'P' to start capturing NEUTRAL face (50 samples)")
print("step 2: press 'P' again to capture BIGGEST SMILE (50 samples)")
print("step 3: enjoy! press 'q' to quit ^^")
print()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("failed to grab frame >_<")
        break
    
    # flip frame horizontally
    frame = cv2.flip(frame, 1)
    
    # convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # process the frame
    results = face_mesh.process(rgb_frame)
    
    # get frame dimensions
    h, w, _ = frame.shape
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # get raw metrics
            width_metric, lift_metric = calculate_raw_metrics(face_landmarks.landmark, w, h)
            
            # store calibration data
            if calibration_state == CALIBRATE_NEUTRAL:
                calibration_data['neutral_width'].append(width_metric)
                calibration_data['neutral_lift'].append(lift_metric)
                if len(calibration_data['neutral_width']) >= 50:
                    print(f"neutral face captured! (50 samples)")
                    print("now press 'P' to capture your BIGGEST SMILE ^^")
                    calibration_state = WAIT_FOR_SMILE
            elif calibration_state == CALIBRATE_SMILE:
                calibration_data['smile_width'].append(width_metric)
                calibration_data['smile_lift'].append(lift_metric)
                if len(calibration_data['smile_width']) >= 50:
                    print(f"smile captured! (50 samples)")
                    if finalize_calibration():
                        calibration_state = CALIBRATED
            
            # calculate smile score
            smile_score, debug_info = calculate_smile_score(face_landmarks.landmark, w, h, debug=DEBUG_MODE)
            
            # draw face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            
            # highlight mouth landmarks
            mouth_landmarks = [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM]
            for landmark_idx in mouth_landmarks:
                landmark = face_landmarks.landmark[landmark_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
            
            # display calibration instructions or smile score
            if calibration_state == WAIT_FOR_START:
                cv2.putText(frame, "press 'P' to start capturing NEUTRAL face", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            elif calibration_state == CALIBRATE_NEUTRAL:
                cv2.putText(frame, "calibration: hold NEUTRAL face...", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"samples: {len(calibration_data['neutral_width'])}/50", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif calibration_state == WAIT_FOR_SMILE:
                cv2.putText(frame, "press 'P' again to capture BIGGEST SMILE", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            elif calibration_state == CALIBRATE_SMILE:
                cv2.putText(frame, "calibration: hold BIGGEST SMILE!", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"samples: {len(calibration_data['smile_width'])}/50", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:  # CALIBRATED
                # change color based on score
                if smile_score < 30:
                    color = (128, 128, 128)
                    text = "neutral"
                elif smile_score < 60:
                    color = (0, 255, 255)
                    text = "slight smile"
                else:
                    color = (0, 255, 0)
                    text = "big smile!"
                
                cv2.putText(frame, f"smile score: {smile_score}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(frame, text, 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # draw debug info
            if DEBUG_MODE:
                y_offset = 200
                line_height = 30
                cv2.putText(frame, f"width_metric: {debug_info.get('width_metric', 0):.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"lift_metric: {debug_info.get('lift_metric', 0):.4f}", 
                           (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if calibration_state == CALIBRATED:
                    cv2.putText(frame, f"width_score: {debug_info.get('width_score', 0):.1f}", 
                               (10, y_offset + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"lift_score: {debug_info.get('lift_score', 0):.1f}", 
                               (10, y_offset + line_height*3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "no face detected >_<", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # show the frame
    cv2.imshow(window_name, frame)
    
    # handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p') or key == ord('P'):
        if calibration_state == WAIT_FOR_START:
            print("capturing neutral face... hold still!")
            calibration_state = CALIBRATE_NEUTRAL
        elif calibration_state == WAIT_FOR_SMILE:
            print("capturing smile... hold that smile!")
            calibration_state = CALIBRATE_SMILE

# cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()