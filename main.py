import cv2
import mediapipe as mp
import numpy as np
import time

 
#Initialize MediaPipe
 

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

 
# Drawing settings
 

# BGR colour format
colors = [
    (0, 0, 255),      # Red
    (255, 0, 255),    # Purple
    (255, 0, 0),      # Blue
    (0, 255, 0)       # Green
]

color_names = ["RED", "PURPLE", "BLUE", "GREEN"]

state = {
    "canvas": None,
    "color_index": 0,
    "eraser": False,
    "previous_point": None
}

# Drawing thickness
brush_size = 15
eraser_size = 40

 
# On-screen button positions
  

buttons = {
    "CLEAR": (10, 10, 95, 55),
    "SAVE": (105, 10, 185, 55),
    "ERASER": (195, 10, 285, 55)
}

# Four colour buttons
color_buttons = []

for i in range(4):
    x1 = 300 + i * 55
    color_buttons.append(
        (x1, 10, x1 + 42, 55)
    )

 
# Mouse click function
 

def mouse_click(event, x, y, flags, param):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # CLEAR button
    x1, y1, x2, y2 = buttons["CLEAR"]

    if x1 <= x <= x2 and y1 <= y <= y2:

        if state["canvas"] is not None:
            state["canvas"][:] = 0

        state["previous_point"] = None
        print("Canvas cleared!")
        return

    # SAVE button
    x1, y1, x2, y2 = buttons["SAVE"]

    if x1 <= x <= x2 and y1 <= y <= y2:

        if state["canvas"] is not None:

            # Create white background
            saved_image = np.full_like(
                state["canvas"], 255
            )

            # Copy drawing pixels onto white background
            mask = np.any(
                state["canvas"] != 0, axis=2
            )

            saved_image[mask] = state["canvas"][mask]

            filename = "air_drawing.png"

            cv2.imwrite(filename, saved_image)

            print("Drawing saved:", filename)

        return

    # ERASER button
    x1, y1, x2, y2 = buttons["ERASER"]

    if x1 <= x <= x2 and y1 <= y <= y2:

        state["eraser"] = not state["eraser"]
        state["previous_point"] = None

        print(
            "Eraser enabled"
            if state["eraser"]
            else "Drawing mode enabled"
        )

        return

    # COLOUR buttons
    for i, (x1, y1, x2, y2) in enumerate(color_buttons):

        if x1 <= x <= x2 and y1 <= y <= y2:

            state["color_index"] = i
            state["eraser"] = False
            state["previous_point"] = None

            print("Selected:", color_names[i])
            return


 
# Draw toolbar
 

def draw_toolbar(frame):

    # Toolbar background
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 70),
        (40, 40, 40),
        -1
    )

    # CLEAR, SAVE and ERASER buttons
    for name, (x1, y1, x2, y2) in buttons.items():

        button_color = (80, 80, 80)

        if name == "ERASER" and state["eraser"]:
            button_color = (0, 100, 200)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            button_color,
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            name,
            (x1 + 8, y1 + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # Four colour buttons
    for i, (x1, y1, x2, y2) in enumerate(color_buttons):

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            colors[i],
            -1
        )

        # Highlight selected colour
        thickness = 3 if i == state["color_index"] else 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            thickness
        )


 
# Finger detection
 

def get_finger_status(landmarks):

    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y

    return index_up, middle_up


 
#Start webcam
 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

cv2.namedWindow("Air Drawing")
cv2.setMouseCallback("Air Drawing", mouse_click)

last_timestamp = 0

with HandLandmarker.create_from_options(options) as detector:

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        if state["canvas"] is None:

            state["canvas"] = np.zeros(
                (height, width, 3),
                dtype=np.uint8
            )

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = max(
            int(time.monotonic() * 1000),
            last_timestamp + 1
        )

        last_timestamp = timestamp

        result = detector.detect_for_video(
            mp_image,
            timestamp
        )

        mode = "NO HAND"
 
         
        #Detect hand and fingers
         

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            index_up, middle_up = get_finger_status(hand)

            # Index fingertip position
            x = int(hand[8].x * width)
            y = int(hand[8].y * height)

            # Draw hand landmarks
            for landmark in hand:

                cx = int(landmark.x * width)
                cy = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (cx, cy),
                    4,
                    (0, 255, 0),
                    -1
                )

            # DRAWING MODE
            if index_up and not middle_up:

                if y > 75:

                    mode = (
                        "ERASING"
                        if state["eraser"]
                        else "DRAWING"
                    )

                    current_point = (x, y)

                    # Select drawing or eraser colour
                    draw_color = (
                        (0, 0, 0)
                        if state["eraser"]
                        else colors[state["color_index"]]
                    )

                    thickness = (
                        eraser_size
                        if state["eraser"]
                        else brush_size
                    )

                    if state["previous_point"] is not None:

                        cv2.line(
                            state["canvas"],
                            state["previous_point"],
                            current_point,
                            draw_color,
                            thickness,
                            cv2.LINE_AA
                        )

                    state["previous_point"] = current_point

                    cv2.circle(
                        frame,
                        current_point,
                        12,
                        draw_color,
                        -1
                    )

                else:

                    state["previous_point"] = None

            # POINTER MODE
            elif index_up and middle_up:

                mode = "POINTER"
                state["previous_point"] = None

                cv2.circle(
                    frame,
                    (x, y),
                    10,
                    (0, 255, 255),
                    -1
                )

            else:

                mode = "STOPPED"
                state["previous_point"] = None

        else:

            state["previous_point"] = None

        # --------------------------------
        # 9. Combine canvas and webcam
        # --------------------------------

        output = cv2.addWeighted(
            frame, 0.7,
            state["canvas"], 1.5,
            0
        )

        # Draw toolbar
        draw_toolbar(output)

        # Display current mode
        cv2.putText(
            output,
            "MODE: " + mode,
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Air Drawing", output)

        # Exit using Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


 
# Release resources
 

cap.release()
cv2.destroyAllWindows()
