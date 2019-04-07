from keras import backend as K
import shutil
from tensorflow.python.client import device_lib
from keras.preprocessing.image import ImageDataGenerator
import os
import json
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from random import sample
from keras import regularizers
from keras.models import model_from_json
import sys

os.chdir('..')
current_dir = os.getcwd()
images_path = os.path.join(current_dir, 'bdd100k_images')
images_path = os.path.join(images_path, 'bdd100k')
images_path = os.path.join(images_path, 'images')
images_path = os.path.join(images_path, '100k')
train_images_path = os.path.join(images_path, 'train')
test_images_path = os.path.join(images_path, 'test')
val_images_path = os.path.join(images_path, 'val')

labels_path = os.path.join(current_dir, 'bdd100k_labels_release')
labels_path = os.path.join(labels_path, 'bdd100k')
labels_path = os.path.join(labels_path, 'labels')
train_labels_path = os.path.join(labels_path, 'bdd100k_labels_images_train.json')
val_labels_path = os.path.join(labels_path, 'bdd100k_labels_images_val.json')

train_dir = os.path.join(current_dir, 'train')
test_dir = os.path.join(current_dir, 'test')
val_dir = os.path.join(current_dir, 'val')

train_dir_size = 50000
val_dir_size = 5000
test_dir_size = 5000
train_size = 5000
val_size = 1000
test_size = 1000


def preprocess():
	K.tensorflow_backend._get_available_gpus()
	#print(device_lib.list_local_devices())

	print("Making train, test, and val directories")
	make_category_dirs(train_dir)
	make_category_dirs(test_dir)
	make_category_dirs(val_dir)

	train_labels = get_labels_file(train_labels_path)
	temp_val_labels = get_labels_file(val_labels_path)
	val_labels = temp_val_labels[0:val_dir_size]
	test_labels = temp_val_labels[val_dir_size:val_dir_size + test_dir_size]

	print("Categorizing train, test and val directories");
	categorize_images(train_labels, train_dir, train_size, train_images_path, train_dir_size)
	categorize_images(test_labels, test_dir, test_size, val_images_path, test_dir_size)
	categorize_images(val_labels, val_dir, val_size, val_images_path, val_dir_size)

def get_labels_file(labels_path):
	file_object = open(labels_path)
	if(file_object == None):
	    print("Could not find labels file")

	print('Found labels file')

	print('Loading json files')
	return json.load(file_object)

def make_dir(dir):
    if(os.path.exists(dir)):
        shutil.rmtree(dir)
    os.mkdir(dir)
    print('Created director ', dir)

def make_category_dirs(dir):
    make_dir(dir)
    low_traffic_dir = os.path.join(dir, 'low')
    make_dir(low_traffic_dir)
    medium_traffic_dir = os.path.join(dir, 'medium')
    make_dir(medium_traffic_dir)

def categorize_images(labels, save_dir, save_dir_size, images_dir, image_dir_size):
    low_dir = os.path.join(save_dir, 'low')
    medium_dir = os.path.join(save_dir, 'medium')
    low_traffic_max_thresh = 4 
    low_traffic_num = 0
    medium_traffic_num = 0
    print('Randomizing labels')
    randomozed_labels = sample(labels, image_dir_size)
    print('Categorizing images')
    for image_value in randomozed_labels:
        image_name = image_value['name']
        image_labels = image_value['labels']
        #print("image is ", image_name)
        drivable_areas = getDrivableArea(image_labels)
        car_count = 0;
        for label in image_labels:
            if isVehicle(label, drivable_areas):
            	car_count = car_count + 1
        src = os.path.join(images_dir, image_name)
        if low_traffic_num > save_dir_size or medium_traffic_num > save_dir_size:
        	break
        if car_count < low_traffic_max_thresh and low_traffic_num < save_dir_size:
            shutil.copy(src, low_dir)
            low_traffic_num = low_traffic_num + 1
        elif car_count > low_traffic_max_thresh and medium_traffic_num < save_dir_size:
        	shutil.copy(src, medium_dir)
        	medium_traffic_num = medium_traffic_num + 1
        #print("image: ", image_name, ' has car count ', car_count)
        #sys.exit()
    print('Categorized ', save_dir)
    print('low traffic no: ', low_traffic_num)
    print('medium traffic no: ', medium_traffic_num)

def getDrivableArea(image_labels):
	drivable_areas = []
	for label in image_labels:
		category = label['category']
		if category == 'drivable area':
			attr = label['attributes']
			area_type = attr['areaType']
			poly2d = label['poly2d']
			vertices = poly2d[0]
			vertices = vertices['vertices']
			drivable_areas.append(vertices)

	return drivable_areas

def inDrivableArea(drivable_areas, box2d):
	x1_car = box2d['x1']
	x2_car = box2d['x2']
	in_left_bounds = False
	in_right_bounds = False
	# Determines if car is in left hand lane
	for area in drivable_areas:
		for vertex in area:
			x_vertex = vertex[0]
			if x1_car > x_vertex and x2_car > x_vertex:
				in_left_bounds = True
				break
	for area in drivable_areas:
		for vertex in area:
			x_vertex = vertex[0]
			if x2_car < x_vertex and x2_car < x_vertex:
				in_right_bounds = True
				break
	return in_left_bounds and in_right_bounds			

def isClose(box2d):
	y_thresh = 40
	y1 = box2d['y1']
	y2 = box2d['y2']
	#print("y1 is {}", y1)
	#print("y2 is {}", y2)
	y_size = y2 - y1
	if(y_size <= y_thresh):
		return False
	return True

def isVehicle(label, drivable_areas):
	category = label['category']
	close = False
	drivable = False
	if category == 'car' or category == 'truck' or category == 'bus':
		box = label['box2d']
		close = isClose(box)
		drivable = inDrivableArea(drivable_areas, box)
		#if close == False:
		#	print(category, ' is to far away')
		#if drivable == False:
		#	print(category, ' is not in a drivable area')
		return close and drivable
	return False

def copy_images(save_dir, images_dir):
    filenames = os.listdir(images_dir)[0:500]
    for filename in filenames:
        src = os.path.join(images_dir, filename)
        shutil.copy(src, save_dir)

def train_network():
	batch_no = 15;
	train_datagen = ImageDataGenerator(rescale=1./255)
	test_datagen = ImageDataGenerator(rescale=1./255)
	val_datagen = ImageDataGenerator(rescale=1./255)

	print('Creating image generators')
	train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_no, class_mode='binary', shuffle=True)
	val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=batch_no, class_mode='binary', shuffle=True)
	test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=batch_no, class_mode='binary', shuffle=True)
	#train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_no, class_mode='categorical', shuffle=True)
	#val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=batch_no, class_mode='categorical', shuffle=True)
	test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=batch_no, class_mode='binary', shuffle=True)
	#for image_batch, label_batch in train_generator:
	 #   print('data batch shape:', image_batch.shape)
	  #  print('labels batch shape:', label_batch.shape)
	  #  break;

	print('Creating network model')

	'''
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(256, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.summary()
	'''
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(256, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(256, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.summary()
	

	'''
	model = models.Sequential()
	model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(3, activation='sigmoid'))
	model.summary()
	'''

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

	train_epoch_steps = train_size / batch_no
	val_epoch_steps = val_size / batch_no
	print("Training network")
	history = model.fit_generator(train_generator, steps_per_epoch=train_epoch_steps, epochs=16, validation_data=val_generator, validation_steps=val_epoch_steps)

	test_epoch_steps = test_size / batch_no
	results = model.evaluate_generator(generator=test_generator, steps=test_epoch_steps)

	print('Results\n loss: {}\n acc: {}', results[0], results[1])

	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")

	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.legend()
	plt.figure()

	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.legend()

	plt.show()

# Program starts here
if len(sys.argv) == 0:
	print('Must have mode argument: preprocess or train')
command = sys.argv[1]
if command == 'process':
	preprocess()
elif command == 'train':
	train_network()
else:
	print('Invalid argument')

