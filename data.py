import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt #for plotting images
import pickle #for saving data


#initialize mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hand = mp_hands.Hands(static_image_mode=True,
                      min_detection_confidence=0.3)

#sets data directory
DATA_DIR = './data'

#creates array to hold xy coords and labels
data = []
labels = []

for dir_ in os.listdir(DATA_DIR): #for each letter directory
    #iterate through images in directory
    for path in os.listdir(os.path.join(DATA_DIR, dir_)): 

        data_secondary = [] #holds xy coords for current image

        #load image and convert to RGB
        image_path = os.path.join(DATA_DIR, dir_, path)
        rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        #process image to extract hand landmarks
        results = hand.process(rgb)
        if results.multi_hand_landmarks:
        #iterates through results, draws landmarks on hand
            for hand_landmarks in results.multi_hand_landmarks:

                #visual drawing of landmarks, uncomment to see
                #make sure to change it to only draw on rgb image once, or else cpu will be tortured
                """
                mp_drawing.draw_landmarks(
                    rgb, #img
                    hand_landmarks, #landmarks
                    mp_hands.HAND_CONNECTIONS, #connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                    )
                """

                for i in range(len(hand_landmarks.landmark)):
                    x =  hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    #finds xyz but we only need xy for 2d image
                    data_secondary.append(x)
                    data_secondary.append(y)

        #data holds subarray of xy coords for each image
        data.append(data_secondary)
        #labels hold the corresponding letter for each image
        labels.append(dir_) 

#data to train classifier on
f = open('hand_data.pkl', 'wb') #creates pickle file to save data
pickle.dump({'data': data, 'labels': labels},f) #dumps data and labels into pickle file
f.close()

        #uncomment below to see images with landmarks drawn

        #plt.figure()
        #plt.imshow(rgb)
        
#plt.show()
