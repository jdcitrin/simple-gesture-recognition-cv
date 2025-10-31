import cv2
import mediapipe as mp
import pickle 
import numpy as np

model_path = pickle.load(open('./model.p', 'rb')) #loads trained model
model = model_path['model'] #extracts model from pickle file

cap = cv2.VideoCapture(0)

#initialize mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#set up hands module
hand = mp_hands.Hands(static_image_mode=False, #treats it as video stream
                      max_num_hands=2, #allow 2 hands in image
                      min_detection_confidence=0.3
                      )

label_dict = {
     #for each item you took images of, add labels here
     0: 'peace',
     1: 'good',
     2: 'middle'

}
while True:

    

    ret, frame = cap.read()
    if not ret:
        break

    #get frame dimensions
    height, width, channel = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process image to extract hand landmarks
    results = hand.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_secondary = [] #holds xy coords for current frame
            xr = []
            yr = []
            mp_drawing.draw_landmarks(
                        rgb, #img
                        hand_landmarks, #landmarks
                        mp_hands.HAND_CONNECTIONS, #connections
                        #styling
                        mp_drawing_styles.get_default_hand_landmarks_style(), 
                        mp_drawing_styles.get_default_hand_connections_style() 
                        )
            for i in range(len(hand_landmarks.landmark)):
                    #get x and y coordinates
                    x =  hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    #finds xyz but we only need xy for 2d image
                    #appends to data array and bbox arrays
                    data_secondary.append(x)
                    data_secondary.append(y)
                    xr.append(x)
                    yr.append(y)

            #calculates bbox coords
            x1 = int(min(xr) * width) - 10
            y1 = int(min(yr) * height) - 10
            x2 = int(max(xr) * width) - 10
            y2 = int(max(yr) * height) - 10

            pred = model.predict([np.asarray(data_secondary)]) #makes prediction
            pred_char = label_dict[int(pred[0])] #gets corresponding letter

            print('predicted letter: {}'.format(pred_char))


        
        cv2.rectangle(rgb, (x1,y1), (x2,y2), (0,0,255), 5) 

        rgb = cv2.putText(
                        img=rgb,                      
                        text='PREDICTED: {}'.format(pred_char),
                        org=(x1,y1-10),                  
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=(0,0,255),
                        thickness=3
                    )


    display_frame = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', display_frame)
    cv2.waitKey(10)



cap.release()
cv2.destroyAllWindows()
