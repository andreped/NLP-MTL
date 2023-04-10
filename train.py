from utils import DatasetLoader
from tqdm import tqdm
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
from models import get_model
from metrics import recall_m, precision_m, f1_m


def main():
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

	# extract features using simple word count
	vectorizer = HashingVectorizer(n_features=5000) #CountVectorizer(max_features=5000) #HashingVectorizer(n_features=5000) # #TfidfVectorizer(max_features=5000)
	X = vectorizer.fit_transform(review_texts)

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

	# define model
	model = get_model(num_feats=X.shape[1], [nb_classes1, nb_classes2, nb_classes3])

	# compile
	model.compile(
		loss="categorical_crossentropy",
		optimizer="adadelta", #Adam(lr=1e-3),
		metrics=["acc", f1_m, precision_m, recall_m]
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

	# evaluate both classifier nets based on the same model
	preds1 = []
	preds2 = []
	preds3 = []
	for i in range(X_test.shape[0]):
		tmp = model.predict(X_test[i])
		preds1.append(np.argmax(tmp[0]))
		preds2.append(np.argmax(tmp[1]))
		preds3.append(np.argmax(tmp[2]))

	y1_orig_test = y1_orig[int(val2 * N):]
	y2_orig_test = y2_orig[int(val2 * N):]
	y3_orig_test = y3_orig[int(val2 * N):]

	print("##### Results #####")
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


if __name__ == "__main__":
	main()
