import pickle 
import numpy as np

#imports ml library/model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#opens pkl file
fulldata = pickle.load(open('./hand_data.pkl', 'rb'))

data = np.asarray(fulldata['data']) #xy coords in numpy array
label = np.asarray(fulldata['labels']) #corresponding labels

# calling tts, splitting data and labels into 2 sets, train and test
#splitting info 80% train, 20% test.
x_train, x_test, y_train, y_test = train_test_split(
    data,
    label, 
    test_size = 0.2, #common split size
    shuffle=True, #shuffle to prevent bias
    stratify=label #split set, maintain label proportions 1/n of data for each label
    )


model = RandomForestClassifier() #creates model
#bootstraps data, creates decision trees on random subsets of data
#then aggregates results, returns most common result.
#reduces correlation between trees, improves accuracy.

model.fit(x_train, y_train) #trains model on training data

model_pred = model.predict(x_test) #makes predictions on test data

acc = accuracy_score(model_pred, y_test) #compares predictions to  labels

print('{}% of samples classified correctly'.format(acc*100)) #prints accuracy

f = open('model.p', 'wb') #creates pickle file to save data
pickle.dump({'model': model},f) #dumps model into pickle file
f.close()