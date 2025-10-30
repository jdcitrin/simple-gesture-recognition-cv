import os
import cv2



#creates data directory if it doesn't exist
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


class_num = 3 #num classes
data_size = 50 #num images per class

cap = cv2.VideoCapture(0) #select cam
for i in range(class_num):
    
    #creates directory for each gesture class
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i)))

    print('collecting data for class {}'.format(i))

    complete = False
    while True:
        #display instruction
        ret , frame = cap.read()
        frame = cv2.putText(
                    img=frame,                      
                    text='press "r" to begin collecting data, ',
                    org=(100,50),                  
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5,
                    color=(0,255,0),
                    thickness=3
                )
        cv2.imshow('frame', frame)



        if cv2.waitKey(10) & 0xFF == ord('r'):
            break

        
    #collects data for class
    count = 0
    while count < data_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(10) 
        cv2.imwrite(os.path.join(DATA_DIR, str(i), '{}.jpg'.format(count)), frame)
        #takes picture every 10 ms and saves to dataset.

        #does 50 times
        count += 1


cap.release()
cv2.destroyAllWindows()