from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense
from keras.utils import to_categorical
try:
    from scipy.ndimage import imread, imresize
except ImportError:
    from scipy.misc import imread, imresize
from keras_status_callback import StatusCallback
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv('.env', verbose=True)


base_model = VGG16(weights='imagenet', include_top=False,
        input_shape=(224, 224, 3))
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(4096, activation='relu'))
add_model.add(Dense(4096, activation='relu'))
add_model.add(Dense(2, activation='softmax'))

for layer in base_model.layers:
    layer.trainable = False

model = Model(inputs=base_model.input,
        outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer='sgd',
        metrics=['accuracy'])


good_files = ['examples/images/cat_resized.jpg']
defect_files = ['examples/images/dog_resized.jpg']

X = np.array([imresize(imread(fname), (224, 224)) for fname in good_files + defect_files])
y = to_categorical(np.array([0, 1]))

user = os.environ['TEST_USER']
db = os.environ['TEST_DB']
host = os.environ['DB_HOST']
password = os.environ['TEST_PASSWORD']

callback = StatusCallback(run_id=0,
        db_connection_string=f'postgres+psycopg2://{user}:{password}@{host}/{db}',
        grayscale=False,
        undersampling=False,
        verbose=True, reset=True)

callback.set_data(
        'training',
        X, y,
        good_files,
        defect_files)
callback.set_data(
        'test',
        None, None,
        None,
        None)
callback.set_data(
        'validation',
        None, None, None, None)

model.fit(X, y, epochs=2, callbacks=[callback, ])
