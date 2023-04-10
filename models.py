from keras.layers import Input, Dense, Dropout
from keras.models import Model


def get_model(num_feats, class_nbs):
	x = Input(shape=(num_feats, ))
	shared = Dense(64, activation="relu")(x) # x
	sub1 = Dense(32, activation="relu")(shared)
	#sub1 = Dropout(0.5)(sub1)

	sub2 = Dense(32, activation="relu")(shared)
	#sub2 = Dropout(0.5)(sub2)

	sub3 = Dense(32, activation="relu")(shared)
	#sub3 = Dropout(0.5)(sub3)

	out1 = Dense(class_nbs[0], activation="softmax")(sub1)
	out2 = Dense(class_nbs[1], activation="softmax")(sub2)
	out3 = Dense(class_nbs[2], activation="softmax")(sub3)

	model = Model(inputs=x, outputs=[out1, out2, out3])
