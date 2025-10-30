import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt #for plotting images


#initialize mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hand = mp_hands.Hands(static_image_mode=True,
                      min_detection_confidence=0.3)


#iterate through images and extract hand landmarks
DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR): #for each letter directory
    #iterate through images in directory
    for path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]: #for now just first image

        #load image and convert to RGB
        image_path = os.path.join(DATA_DIR, dir_, path)
        rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        #process image to extract hand landmarks
        results = hand.process(rgb)
        if results.multi_hand_landmarks:
        #iterates through results, draws landmarks on hand
            for hand_landmarks in results.multi_hand_landmarks:

                #visual drawing of landmarks, uncomment to see

                mp_drawing.draw_landmarks(
                    rgb, #img
                    hand_landmarks, #landmarks
                    mp_hands.HAND_CONNECTIONS, #connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                    )
                


        plt.figure()
        plt.imshow(rgb)
        
plt.show()
