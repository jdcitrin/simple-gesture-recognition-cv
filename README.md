Working real time hand gesture recognition program that takes in hand gestures through landmarks, uses a random forest model to generate predictions based on landmark x's and y's, and outputs what gesture you are making based on data set provided by users previously provided data set.


to make venv: python -m venv venv

to enter virtual environment: source venv/bin/activate 

to get necessary addons: pip install -r requirements.txt

STEPS TO OPERATE:

1. run $ python ./cam.py 
    through this program, you are feeding it the live images for the data set to create predictions out of. you can change the number of distinct gestures you want to input, and the number of images to train model for each gesture for increased accuracy. it will take images of you for each gesture. You will see the images in /data 

2. run $ python ./data.py
    running this attaches landmarks to the hands, and then feeds the xy coordinates of the landmark points into an array to be used in training the model. It outputs these arrays into a pickle file, both the xy data and the labels attached to them.
    you can find these in hand_data.pkl

3. run $ python ./classifier.py
    running this uses the data provided in the pkl file to train the random forest model, which uses multiple randomized decision trees with x's and y's from the landmarks in order to generate probabilities to predict specific gestures. Trains on 80% of the data provided, and tests on 20%, usually generating a 100% accuracy. This also outputs a pickle file, with the models data to live predict the gestures.

4. run $ python ./test-classifier.py
    this is the final portion of the program, using the xy coordinates from real time landmark detection, placing it into an array, using the data to make a prediction on the gesture made, and outputting it above the bbox surrounding the hand. You must manually set the previously inputted gestures to whatever names you desire for them, from 0:n gestures you inputted. 



:D