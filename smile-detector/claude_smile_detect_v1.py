import cv2
import mediapipe as mp
import numpy as np

# initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# mouth landmark indices
# left corner: 61, right corner: 291
# upper lip center: 13, lower lip center: 14
# top lip: 0, bottom lip: 17
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
UPPER_LIP = 0
LOWER_LIP = 17
FOREHEAD_TOP = 10  # top of forehead
CHIN_BOTTOM = 152  # bottom of chin

def calculate_distance(point1, point2):
    """calculate euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_smile_score(landmarks, image_width, image_height, debug=False):
    """
    calculate smile score from 0-100 based on mouth shape
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

    # calculate mouth width (horizontal distance between corners)
    mouth_width = calculate_distance([left_x, left_y], [right_x, right_y])
    
    # calculate mouth height (vertical distance)
    mouth_height = abs(bottom_y - top_y)
    
    # calculate how much the corners are lifted (smile indicator)
    # compare the y position of corners vs center
    center_y = (top_y + bottom_y) / 2
    avg_corner_y = (left_y + right_y) / 2
    corner_lift = center_y - avg_corner_y  # positive = corners lifted up
    
    # calculate width-to-height ratio (wider = more smile)
    if mouth_height > 0:
        aspect_ratio = mouth_width / mouth_height
    else:
        aspect_ratio = 0
    
    # scoring factors
    # typical neutral mouth: aspect_ratio ~3-4
    # big smile: aspect_ratio ~6-8, corners lifted
    
    # normalize aspect ratio (neutral at 3.5, smile at 7+)
    ratio_score = (aspect_ratio - 3.5) / 4.0 * 70

    # width score, neutral ~50, big smile ~100
    mouth_width_to_face_height = mouth_width / face_height * 300
    width_score = (mouth_width_to_face_height - 50) / 50 * 50
    
    # normalize corner lift (scale it relative to face size)
    corner_lift_to_face_height = corner_lift / face_height
    lift_score = corner_lift_to_face_height * 9000
    
    # combine scores
    smile_score = width_score + lift_score
    
    # return debug info if requested
    if debug:
        return int(smile_score), {
            'center_y': center_y,
            'avg_corner_y': avg_corner_y,
            'corner_lift': corner_lift,
            'aspect_ratio': aspect_ratio,
            'ratio_score': ratio_score,
            'lift_score': lift_score,
            'mouth_width_to_face_height': mouth_width_to_face_height,
            'width_score': width_score
        }
    
    return int(smile_score)

# start webcam
cap = cv2.VideoCapture(0)

# debug mode toggle - set to True to see variables
DEBUG_MODE = True

print("smile detector started! press 'q' to quit ^^")
print(f"debug mode: {DEBUG_MODE}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("failed to grab frame >_<")
        break
    
    # flip frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)
    
    # convert to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # process the frame
    results = face_mesh.process(rgb_frame)
    
    # get frame dimensions
    h, w, _ = frame.shape
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # calculate smile score
            if DEBUG_MODE:
                smile_score, debug_info = calculate_smile_score(face_landmarks.landmark, w, h, debug=True)
            else:
                smile_score = calculate_smile_score(face_landmarks.landmark, w, h)
            
            # draw face mesh (optional, makes it look cool)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            
            # highlight the specific mouth landmarks we're using in purple
            mouth_landmarks = [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM]
            for landmark_idx in mouth_landmarks:
                landmark = face_landmarks.landmark[landmark_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)  # purple filled circle
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)  # white outline
            
            # display smile score
            # change color based on score
            if smile_score < 30:
                color = (128, 128, 128)  # gray
                text = "neutral"
            elif smile_score < 60:
                color = (0, 255, 255)  # yellow
                text = "slight smile"
            else:
                color = (0, 255, 0)  # green
                text = "big smile!"
            
            # draw score on screen
            cv2.putText(frame, f"smile score: {smile_score}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, text, 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # draw debug info if enabled
            if DEBUG_MODE:
                y_offset = 150
                line_height = 35
                cv2.putText(frame, f"center_y: {debug_info['center_y']:.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"avg_corner_y: {debug_info['avg_corner_y']:.2f}", 
                           (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"corner_lift: {debug_info['corner_lift']:.2f}", 
                           (10, y_offset + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"aspect_ratio: {debug_info['aspect_ratio']:.2f}", 
                           (10, y_offset + line_height*3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"ratio_score: {debug_info['ratio_score']:.2f}", 
                           (10, y_offset + line_height*4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"lift_score: {debug_info['lift_score']:.2f}", 
                           (10, y_offset + line_height*5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"mouth_width_to_face_height: {debug_info['mouth_width_to_face_height']:.2f}", 
                           (10, y_offset + line_height*6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"width_score: {debug_info['width_score']:.2f}", 
                           (10, y_offset + line_height*7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "no face detected", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # show the frame
    cv2.imshow('smile detector ^^', frame)
    
    # quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()