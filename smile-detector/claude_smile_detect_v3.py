"""
Pose-invariant smile detection using mediapipe face mesh.

This function computes a pose-invariant smile metric by rotating mediapipe's 3D
face landmarks into a canonical face coordinate frame and measuring mouth width
relative to a stable facial reference (inter-ocular distance).

Based on chatgpt_recs.md specification.
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# landmark indices
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_EYE_OUTER = 33      # left eye outer corner
RIGHT_EYE_OUTER = 263    # right eye outer corner
NOSE_TIP = 4
NOSE_BRIDGE = 6          # top of nose bridge (between eyes)

# calibration states
WAIT_FOR_START = 0
CALIBRATE_NEUTRAL = 1
WAIT_FOR_SMILE = 2
CALIBRATE_SMILE = 3
CALIBRATED = 4

# calibration data
calibration_state = WAIT_FOR_START
calibration_data = {
    'neutral_ratios': [],
    'smile_ratios': [],
    'min_ratio': None,
    'max_ratio': None
}


def get_landmark_3d(landmarks, idx):
    """Extract 3D coordinates from a landmark as numpy array."""
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])


def compute_pose_invariant_smile_ratio(landmarks, print_debug=False):
    """
    Compute a pose-invariant smile metric.

    Steps:
    1. Extract 3D landmarks
    2. Remove translation (anchor at nose tip)
    3. Build canonical face coordinate frame (using rigid landmarks only)
    4. Transform points into face space
    5. Compute smile_width / inter_ocular using X-COMPONENTS ONLY (not 3D norms)

    Returns: ratio (float), debug_info (dict)
    """
    # Stage 1: Extract 3D points
    nose_tip = get_landmark_3d(landmarks, NOSE_TIP)
    nose_bridge = get_landmark_3d(landmarks, NOSE_BRIDGE)
    left_eye_outer = get_landmark_3d(landmarks, LEFT_EYE_OUTER)
    right_eye_outer = get_landmark_3d(landmarks, RIGHT_EYE_OUTER)
    mouth_left = get_landmark_3d(landmarks, MOUTH_LEFT)
    mouth_right = get_landmark_3d(landmarks, MOUTH_RIGHT)

    if print_debug:
        print("\n" + "="*60)
        print("STAGE 1: Raw MediaPipe 3D points")
        print("="*60)
        print(f"  nose_tip      = {nose_tip}")
        print(f"  nose_bridge   = {nose_bridge}")
        print(f"  left_eye      = {left_eye_outer}")
        print(f"  right_eye     = {right_eye_outer}")
        print(f"  mouth_left    = {mouth_left}")
        print(f"  mouth_right   = {mouth_right}")

    # Stage 2: Remove translation (anchor at nose tip)
    anchor = nose_tip
    left_eye_t = left_eye_outer - anchor
    right_eye_t = right_eye_outer - anchor
    nose_bridge_t = nose_bridge - anchor
    mouth_left_t = mouth_left - anchor
    mouth_right_t = mouth_right - anchor

    # Eyes midpoint (for y-axis)
    eyes_mid_t = (left_eye_t + right_eye_t) / 2

    if print_debug:
        print("\n" + "="*60)
        print("STAGE 2: After translation (anchor = nose_tip)")
        print("="*60)
        print(f"  nose_tip_t    = [0, 0, 0]  (anchor)")
        print(f"  nose_bridge_t = {nose_bridge_t}")
        print(f"  left_eye_t    = {left_eye_t}")
        print(f"  right_eye_t   = {right_eye_t}")
        print(f"  mouth_left_t  = {mouth_left_t}")
        print(f"  mouth_right_t = {mouth_right_t}")
        print(f"  eyes_mid_t    = {eyes_mid_t}")

    # Stage 3: Build canonical face coordinate frame
    # X-axis: outer eye corners only (stable under yaw)
    v_x = right_eye_t - left_eye_t
    x_hat = v_x / np.linalg.norm(v_x)

    # Y-axis: nose bridge → nose tip (along the nose, captures pitch well)
    # Since nose_tip is anchor at origin, this is -nose_bridge_t
    v_y = -nose_bridge_t
    y_hat = v_y / np.linalg.norm(v_y)

    # Z-axis: cross product (face normal)
    z_hat = np.cross(x_hat, y_hat)
    z_hat = z_hat / np.linalg.norm(z_hat)

    # Re-orthogonalize Y to ensure orthonormal basis
    y_hat_original = y_hat.copy()
    y_hat = np.cross(z_hat, x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)

    if print_debug:
        print("\n" + "="*60)
        print("STAGE 3: Build face coordinate frame")
        print("="*60)
        print(f"  v_x (right_eye - left_eye) = {v_x}")
        print(f"  x_hat = normalize(v_x)     = {x_hat}")
        print(f"  v_y (nose_bridge → nose_tip) = {v_y}")
        print(f"  y_hat (before orthog)      = {y_hat_original}")
        print(f"  z_hat = cross(x, y)        = {z_hat}")
        print(f"  y_hat (after orthog)       = {y_hat}")
        print(f"\n  x_hat · y_hat = {np.dot(x_hat, y_hat):.6f}  (should be ~0)")
        print(f"  x_hat · z_hat = {np.dot(x_hat, z_hat):.6f}  (should be ~0)")
        print(f"  y_hat · z_hat = {np.dot(y_hat, z_hat):.6f}  (should be ~0)")

    # Rotation matrix: columns are face axes
    R = np.column_stack([x_hat, y_hat, z_hat])

    if print_debug:
        print("\n" + "="*60)
        print("ROTATION MATRIX R (columns = face axes)")
        print("="*60)
        print(f"  R = ")
        print(f"      [{R[0,0]:+.4f}  {R[0,1]:+.4f}  {R[0,2]:+.4f}]")
        print(f"      [{R[1,0]:+.4f}  {R[1,1]:+.4f}  {R[1,2]:+.4f}]")
        print(f"      [{R[2,0]:+.4f}  {R[2,1]:+.4f}  {R[2,2]:+.4f}]")
        print(f"\n  det(R) = {np.linalg.det(R):.6f}  (should be 1)")

    # Stage 4: Transform points into face space
    left_eye_face = R.T @ left_eye_t
    right_eye_face = R.T @ right_eye_t
    mouth_left_face = R.T @ mouth_left_t
    mouth_right_face = R.T @ mouth_right_t
    nose_bridge_face = R.T @ nose_bridge_t

    if print_debug:
        print("\n" + "="*60)
        print("STAGE 4: Points in face space (R.T @ p)")
        print("="*60)
        print(f"  nose_tip_face     = [0, 0, 0]  (anchor)")
        print(f"  nose_bridge_face  = {nose_bridge_face}")
        print(f"  left_eye_face     = {left_eye_face}")
        print(f"  right_eye_face    = {right_eye_face}")
        print(f"  mouth_left_face   = {mouth_left_face}")
        print(f"  mouth_right_face  = {mouth_right_face}")

    # Stage 5: Compute distances using X-COMPONENT ONLY (kills z/y noise)
    inter_ocular_x = abs(right_eye_face[0] - left_eye_face[0])
    smile_width_x = abs(mouth_right_face[0] - mouth_left_face[0])

    # Compute ratio
    ratio = smile_width_x / inter_ocular_x if inter_ocular_x > 0 else 0

    if print_debug:
        print("\n" + "="*60)
        print("STAGE 5: Final measurements (X-component only)")
        print("="*60)
        print(f"  inter_ocular_x = |{right_eye_face[0]:.4f} - {left_eye_face[0]:.4f}| = {inter_ocular_x:.4f}")
        print(f"  smile_width_x  = |{mouth_right_face[0]:.4f} - {mouth_left_face[0]:.4f}| = {smile_width_x:.4f}")
        print(f"  ratio = smile_width_x / inter_ocular_x = {ratio:.4f}")
        print("="*60 + "\n")

    debug_info = {
        'inter_ocular_x': inter_ocular_x,
        'smile_width_x': smile_width_x,
        'ratio': ratio,
        'x_hat': x_hat,
        'y_hat': y_hat,
        'z_hat': z_hat,
        # Translated points (after stage 2, before rotation)
        'translated_points': {
            'left_eye': left_eye_t,
            'right_eye': right_eye_t,
            'mouth_left': mouth_left_t,
            'mouth_right': mouth_right_t,
            'nose_tip': np.array([0, 0, 0]),
            'nose_bridge': nose_bridge_t,
        },
        # Face-space points (after stage 4, fully transformed)
        'face_points': {
            'left_eye': left_eye_face,
            'right_eye': right_eye_face,
            'mouth_left': mouth_left_face,
            'mouth_right': mouth_right_face,
            'nose_tip': np.array([0, 0, 0]),
            'nose_bridge': nose_bridge_face,
        }
    }

    return ratio, debug_info


def linear_map(value, min_val, max_val):
    """Map value from [min_val, max_val] to [0, 100]. Can exceed range."""
    if max_val == min_val:
        return 50
    return (value - min_val) / (max_val - min_val) * 100


def calculate_smile_score(landmarks, debug=False):
    """
    Calculate smile score from 0-100 based on calibration.
    """
    ratio, debug_info = compute_pose_invariant_smile_ratio(landmarks)

    if calibration_state != CALIBRATED:
        return 0, debug_info

    smile_score = linear_map(ratio,
                             calibration_data['min_ratio'],
                             calibration_data['max_ratio'])

    if debug:
        debug_info['smile_score'] = smile_score
        debug_info['min_ratio'] = calibration_data['min_ratio']
        debug_info['max_ratio'] = calibration_data['max_ratio']

    return int(smile_score), debug_info


def finalize_calibration():
    """Calculate min/max from calibration samples."""
    if len(calibration_data['neutral_ratios']) > 0 and len(calibration_data['smile_ratios']) > 0:
        all_ratios = calibration_data['neutral_ratios'] + calibration_data['smile_ratios']

        calibration_data['min_ratio'] = min(all_ratios)
        calibration_data['max_ratio'] = max(all_ratios)

        neutral_avg = np.mean(calibration_data['neutral_ratios'])
        smile_avg = np.mean(calibration_data['smile_ratios'])

        print(f"\ncalibration complete! ^^")
        print(f"ratio range: {calibration_data['min_ratio']:.3f} to {calibration_data['max_ratio']:.3f}")
        print(f"neutral avg: {neutral_avg:.3f}, smile avg: {smile_avg:.3f}\n")
        return True
    return False


# start webcam
cap = cv2.VideoCapture(0)

# create named window
window_name = 'smile detector v3 (pose-invariant) ^^'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

window_width = 1280
window_height = 720
x_pos = (screen_width - window_width) // 2
y_pos = (screen_height - window_height) // 2
cv2.moveWindow(window_name, x_pos, y_pos)

DEBUG_MODE = True
print_vector_debug = False  # set True on 'D' keypress for one frame

print("=== smile detector v3 (pose-invariant) ===")
print("uses 3D face coordinate frame for rotation invariance")
print()
print("step 1: press 'P' to start capturing NEUTRAL face (50 samples)")
print("step 2: press 'P' again to capture BIGGEST SMILE (50 samples)")
print("step 3: enjoy! press 'q' to quit ^^")
print("press 'D' to print vector math debug for one frame")
print()

# Set up matplotlib for live 3D plotting
plt.ion()  # interactive mode
fig = plt.figure(figsize=(14, 6))

# Left plot: Translated points (raw MediaPipe, just centered)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlim(-0.15, 0.15)
ax1.set_ylim(-0.15, 0.15)
ax1.set_zlim(-0.15, 0.15)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Translated (raw MediaPipe, centered)')

# Right plot: Face space points (after rotation)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlim(-0.15, 0.15)
ax2.set_ylim(-0.15, 0.15)
ax2.set_zlim(-0.15, 0.15)
ax2.set_xlabel('X (left-right)')
ax2.set_ylabel('Y (down-up)')
ax2.set_zlabel('Z (in-out)')
ax2.set_title('Face Space (pose-normalized)')

# We'll recreate scatter plots each frame for simplicity with 3D
plt.tight_layout()
plt.show(block=False)

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
            # compute pose-invariant ratio
            ratio, debug_info = compute_pose_invariant_smile_ratio(
                face_landmarks.landmark,
                print_debug=print_vector_debug
            )
            print_vector_debug = False  # reset after one frame

            # update matplotlib 3D plots
            if 'face_points' in debug_info and 'translated_points' in debug_info:
                tp = debug_info['translated_points']
                fp = debug_info['face_points']

                # Clear previous plots
                ax1.cla()
                ax2.cla()

                # Set limits and labels for ax1 (translated)
                ax1.set_xlim(-0.15, 0.15)
                ax1.set_ylim(-0.15, 0.15)
                ax1.set_zlim(-0.15, 0.15)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('Z')
                ax1.set_title('Translated (raw MediaPipe, centered)')

                # Set limits and labels for ax2 (face space)
                ax2.set_xlim(-0.15, 0.15)
                ax2.set_ylim(-0.15, 0.15)
                ax2.set_zlim(-0.15, 0.15)
                ax2.set_xlabel('X (left-right)')
                ax2.set_ylabel('Y (down-up)')
                ax2.set_zlabel('Z (in-out)')
                ax2.set_title('Face Space (pose-normalized)')

                # Plot translated points (ax1)
                for name, color in [('left_eye', 'green'), ('right_eye', 'green'),
                                    ('mouth_left', 'magenta'), ('mouth_right', 'magenta'),
                                    ('nose_tip', 'yellow'), ('nose_bridge', 'cyan')]:
                    p = tp[name]
                    ax1.scatter(p[0], p[1], p[2], c=color, s=100)
                    ax1.text(p[0], p[1], p[2], f'  {name}', fontsize=8)

                # Draw lines connecting eyes and mouth (translated)
                ax1.plot([tp['left_eye'][0], tp['right_eye'][0]],
                        [tp['left_eye'][1], tp['right_eye'][1]],
                        [tp['left_eye'][2], tp['right_eye'][2]], 'g-', linewidth=2)
                ax1.plot([tp['mouth_left'][0], tp['mouth_right'][0]],
                        [tp['mouth_left'][1], tp['mouth_right'][1]],
                        [tp['mouth_left'][2], tp['mouth_right'][2]], 'm-', linewidth=2)

                # Plot face space points (ax2)
                for name, color in [('left_eye', 'green'), ('right_eye', 'green'),
                                    ('mouth_left', 'magenta'), ('mouth_right', 'magenta'),
                                    ('nose_tip', 'yellow'), ('nose_bridge', 'cyan')]:
                    p = fp[name]
                    ax2.scatter(p[0], p[1], p[2], c=color, s=100)
                    ax2.text(p[0], p[1], p[2], f'  {name}', fontsize=8)

                # Draw lines connecting eyes and mouth (face space)
                ax2.plot([fp['left_eye'][0], fp['right_eye'][0]],
                        [fp['left_eye'][1], fp['right_eye'][1]],
                        [fp['left_eye'][2], fp['right_eye'][2]], 'g-', linewidth=2)
                ax2.plot([fp['mouth_left'][0], fp['mouth_right'][0]],
                        [fp['mouth_left'][1], fp['mouth_right'][1]],
                        [fp['mouth_left'][2], fp['mouth_right'][2]], 'm-', linewidth=2)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # store calibration data
            if calibration_state == CALIBRATE_NEUTRAL:
                calibration_data['neutral_ratios'].append(ratio)
                if len(calibration_data['neutral_ratios']) >= 50:
                    print(f"neutral face captured! (50 samples)")
                    print("now press 'P' to capture your BIGGEST SMILE ^^")
                    calibration_state = WAIT_FOR_SMILE
            elif calibration_state == CALIBRATE_SMILE:
                calibration_data['smile_ratios'].append(ratio)
                if len(calibration_data['smile_ratios']) >= 50:
                    print(f"smile captured! (50 samples)")
                    if finalize_calibration():
                        calibration_state = CALIBRATED

            # calculate smile score
            smile_score, score_debug = calculate_smile_score(face_landmarks.landmark, debug=DEBUG_MODE)

            # draw face mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # highlight key landmarks
            key_landmarks = [MOUTH_LEFT, MOUTH_RIGHT, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, NOSE_TIP, NOSE_BRIDGE]
            colors = [(255, 0, 255), (255, 0, 255), (0, 255, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255)]
            for landmark_idx, color in zip(key_landmarks, colors):
                landmark = face_landmarks.landmark[landmark_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 8, color, -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

            # display calibration instructions or smile score
            if calibration_state == WAIT_FOR_START:
                cv2.putText(frame, "press 'P' to start capturing NEUTRAL face",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            elif calibration_state == CALIBRATE_NEUTRAL:
                cv2.putText(frame, "calibration: hold NEUTRAL face...",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"samples: {len(calibration_data['neutral_ratios'])}/50",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif calibration_state == WAIT_FOR_SMILE:
                cv2.putText(frame, "press 'P' again to capture BIGGEST SMILE",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            elif calibration_state == CALIBRATE_SMILE:
                cv2.putText(frame, "calibration: hold BIGGEST SMILE!",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"samples: {len(calibration_data['smile_ratios'])}/50",
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
                cv2.putText(frame, f"ratio: {debug_info.get('ratio', 0):.3f}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"smile_width_x: {debug_info.get('smile_width_x', 0):.3f}",
                           (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"inter_ocular_x: {debug_info.get('inter_ocular_x', 0):.3f}",
                           (10, y_offset + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if calibration_state == CALIBRATED:
                    cv2.putText(frame, f"range: [{score_debug.get('min_ratio', 0):.3f}, {score_debug.get('max_ratio', 0):.3f}]",
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
    elif key == ord('d') or key == ord('D'):
        print_vector_debug = True

# cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
plt.close(fig)
