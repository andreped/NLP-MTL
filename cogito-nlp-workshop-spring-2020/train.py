from utils import DatasetLoader
import keras
from tqdm import tqdm
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
import matplotlib.pyplot as plt 

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def undersample(X, y1):
	counts, tmp_classes = np.histogram(y1, bins=np.unique(y1))
	min_count = min(counts)
	X_new = []
	y_new = []
	for c in tmp_classes:
		tmp = y1[y1 == c]
		order = np.array(range(len(tmp)))
		np.random.shuffle(order)
		tmp = tmp[order]
		X = X[order]
		X_tmp = X[:min_count]
		y_tmp = tmp[:min_count]
		for x, y in zip(X_tmp, y_tmp):
			X_new.append(x)
			y_new.append(y)
	return X_new, y_new


dataset = DatasetLoader.load_reviews()

class_names = []
recommends = []
upvotes = []
review_texts = []
review_rating = []
deps = []
for review in tqdm(dataset):
	class_names.append(review.class_name)
	recommends.append(review.recommends)
	upvotes.append(review.upvotes)
	review_texts.append(review.review_text)
	review_rating.append(review.rating)
	deps.append(review.department_name)

from bert_embedding import BertEmbedding

bert_embedding = BertEmbedding()
X = bert_embedding(review_texts)


# extract features
#vectorizer = HashingVectorizer(n_features=5000) #CountVectorizer(max_features=5000) #HashingVectorizer(n_features=5000) # #TfidfVectorizer(max_features=5000)
#X = vectorizer.fit_transform(review_texts)

y1 = np.array(deps)
y2 = np.array(recommends)#.astype(int)
y3 = np.array(review_rating)

y2 = np.array([str(x) for x in y2])

# filter away all elements with nan
cands = y1 != "nan"

X = X[cands]
y1 = y1[cands]
y2 = y2[cands]
y3 = y3[cands]

# factorize GT
label_encoder = LabelEncoder()
y1 = label_encoder.fit_transform(y1)
classes1 = label_encoder.classes_

label_encoder = LabelEncoder()
y2 = label_encoder.fit_transform(y2)
classes2 = label_encoder.classes_

label_encoder = LabelEncoder()
y3 = label_encoder.fit_transform(y3)
classes3 = label_encoder.classes_

print(np.histogram(y1, bins=np.unique(y1)))
print(np.histogram(y2, bins=np.unique(y2)))
print(np.histogram(y3, bins=np.unique(y3)))

# undersample to balance classes
#X1, y1 = undersample(y1)
#X2, y2 = undersample(y2)
#X3, y3 = undersample(y3)




# remove low represented classes
#conds = (y1 != 3) & (y1 != 5)
#X = X[conds]
#y1 = y1[conds]
#y2 = y2[conds]
#y3 = y3[conds]

# shuffle
order = np.array(range(len(y1)))
np.random.shuffle(order)
X = X[order]
y1 = y1[order]
y2 = y2[order]
y3 = y3[order]

# factorize GT again
label_encoder = LabelEncoder()
y1 = label_encoder.fit_transform(y1)
classes1_new = label_encoder.classes_

label_encoder = LabelEncoder()
y2 = label_encoder.fit_transform(y2)
classes2_new = label_encoder.classes_

label_encoder = LabelEncoder()
y3 = label_encoder.fit_transform(y3)
classes3_new = label_encoder.classes_

# number of classes for each set
nb_classes1 = len(np.unique(y1))
nb_classes2 = len(np.unique(y2))
nb_classes3 = len(np.unique(y3))

y1_orig = y1.copy()
y2_orig = y2.copy()
y3_orig = y3.copy()

# split into train, val and test set (before split) to calc class weights
val1 = 0.8; val2 = 0.9
N = X.shape[0]

X_train = X[:int(val1*N)]
y1_train = y1[:int(val1*N)]
y2_train = y2[:int(val1*N)]
y3_train = y3[:int(val1*N)]

# class weights for each set of GTs
class_weights1 = compute_class_weight("balanced", np.unique(y1_train), y1_train)
class_weights2 = compute_class_weight("balanced", np.unique(y2_train), y2_train)
class_weights3 = compute_class_weight("balanced", np.unique(y3_train), y3_train)

# one-hot encode
y1 = np.eye(nb_classes1)[y1]
y2 = np.eye(nb_classes2)[y2]
y3 = np.eye(nb_classes3)[y3]

# split into train, val and test set
X_train = X[:int(val1*N)]
y1_train = y1[:int(val1*N)]
y2_train = y2[:int(val1*N)]
y3_train = y3[:int(val1*N)]

X_val = X[int(val1*N):int(val2*N)]
y1_val = y1[int(val1*N):int(val2*N)]
y2_val = y2[int(val1*N):int(val2*N)]
y3_val = y3[int(val1*N):int(val2*N)]

X_test = X[int(val2*N):]
y1_test = y1[int(val2*N):]
y2_test = y2[int(val2*N):]
y3_test = y3[int(val2*N):]

N = X.shape[0]
num_feats = X.shape[1]


# model
x = Input(shape=(num_feats, ))
shared = Dense(64, activation="relu")(x) # x
sub1 = Dense(32, activation="relu")(shared)
#sub1 = Dropout(0.5)(sub1)

sub2 = Dense(32, activation="relu")(shared)
#sub2 = Dropout(0.5)(sub2)

sub3 = Dense(32, activation="relu")(shared)
#sub3 = Dropout(0.5)(sub3)

out1 = Dense(nb_classes1, activation="softmax")(sub1)
out2 = Dense(nb_classes2, activation="softmax")(sub2)
out3 = Dense(nb_classes3, activation="softmax")(sub3)

model = Model(inputs=x, outputs=[out1, out2, out3])

# compile
model.compile(
	loss="categorical_crossentropy",
	optimizer="adadelta", #Adam(lr=1e-3),
	metrics=["acc", f1_m,precision_m, recall_m] #["categorical_accuracy"]
	)

# saving best model
mcp_save = ModelCheckpoint(
	'.multi_task_network.hdf5',
	save_best_only=True,
	monitor='val_loss',
	mode='auto',
	)

# fit
model.fit(
	X_train,
	[y1_train, y2_train, y3_train],
	validation_data=(X_val, [y1_val, y2_val, y3_val]),
	epochs=50,
	batch_size=512,
	class_weight=[class_weights1, class_weights2, class_weights3],
	verbose=1
	)

# eval
#_, accuracy = model.evaluate(X_test, [y1_test, y2_test])
#print('Accuracy on test set: %.2f' % (accuracy*100))

# evaluate both classifier nets based on the same model
preds1 = []
preds2 = []
preds3 = []
for i in range(X_test.shape[0]):
	tmp = model.predict(X_test[i])
	preds1.append(np.argmax(tmp[0]))
	preds2.append(np.argmax(tmp[1]))
	preds3.append(np.argmax(tmp[2]))

y1_orig_test = y1_orig[int(val2*N):]
y2_orig_test = y2_orig[int(val2*N):]
y3_orig_test = y3_orig[int(val2*N):]


print("\n\n\n")
print(classes1)
print(classes1_new)
print(np.unique(preds1))
print(np.unique(y1_orig_test))

print()
print(classes2)
print(classes2_new)
print(np.unique(preds2))
print(np.unique(y2_orig_test))

print()
print(classes3)
print(classes3_new)
print(np.unique(preds3))
print(np.unique(y3_orig_test))
print(classes3[classes3_new])

# evaluate model on each task
print("----")
print(classification_report(y1_orig_test, preds1, target_names=classes1[classes1_new]))
print("----")
print(classification_report(y2_orig_test, preds2, target_names=classes2[classes2_new]))
print("----")
print(classification_report(y3_orig_test, preds3))#, target_names=classes3[classes3_new]))

exit()

