import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math


tf.__version__

#Convolutional layer 1
#when dealing with high-dimensional inputs such as images
#it is impractical to connect neurons to all neurons in previous volume
#instead we will connect each neuron to only a local region of the input volume.
#the spatial extent of this connectivity is a hyperparameter called receptive field
#of the neuron(equivalently this is the filter size)
#smaller size than input

filter_size1 = 5 #convolutional filter are 5 x 5 pixels
num_filter1 = 16 #there are 116 of these filters

#more filters feature map will n
#convolutoinal layyes 2
filter_size2 = 5 #convolutional filters are 5 x 5 pixels
num_filter = 36 # There are 36 of these filters

#Fullyy connected layers
fc_size=128 #number of neurons in fully connected layer

#Load Data
from tensorflow.examples.tutorials.mnist import imput_data
data = imput_data.read_data_Sets('data/MNIST/', one_hot =True)

print("size of:")
print("-Training set : \t\t{}".format(len(data.train.labels)))
print("-Training set:\t\t{}".format(len(data.test.labels)))
print("-validation-set:\t\t{}".format(len(data.validation.labels)))


data.test.cls = np.argmax(data.test.labels, aixs =1)

#data dimensions

#we know that MNIST images are 28 pixels in each dimension
img_size = 28

#Image are stored in our-dimnesional arrays of this length
img_size_flat = img_size * img_size

#tuple wuth height and width of images used to reshape arrays
img_shape = (img_size, img_size)

#number of color xhannels for the images: 1 channel for gray scale
num_channels =1 

#number of classes, one class for each digits
nm_classes =10


def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
	
def new_biases(length):
	#equivalent to y intercept
	#constant value carried oer across matrix math
	return tf.Variable(tf.constant(0.05, shape=[length]))
	

#helper function for cerating a new conoultional layer
def new_conv_layer(input,
					num_imput_channels,
					filter_size,
					num_fiilters,
					use_pooling=True):
	
	#shape of filter-weigghts for the convolution
	#this format is determined by Tensorflow API
	shape = [filter_size, filter_size, num_input_channels, num_filters]
					
	#create new weights aka filters with the given shape.
	weights = new_weights(shape=shape)
	
	#create new biases, one for each filter.
	biases = new_biases(length = num_filters)
	
	#create tensorflow operation for convolution.
	#note the strides are set to 1 in all dimensions.
	#the firsts and last stride must always be 1
	#because first is for image number and 
	#last is for the input channel
	#but e.g. strides = [1,2,2,1] would mean that the filter
	#is moved 2 pixels across x and y axis of the image
	#The padding is set to 'SAME' which means the input image
	# is padded with zeros so the size of the output is the same.
	layer = tf.nn.conv2d(input =input,
						filter = weights,
						strides = [1,1,1,1],
						padding='SAME')
						
	#add the biases to the results of the convolution
	#a bias value is added to each filter -channel
	layer+ = biases
	
	#use pooling to down sapmle the image resolution
	if use_pooling:
	#this is 2x2 max pooling, which means that we
	#consider 2x2 windows and select the larget value
	#in each window. Then we move 2 pixels to next window.
		layer = tf.nn.max_pool(value = layer,
							ksize = [1,2,2,1],
							strides=[1,2,2,1],
							padding='SAME')
							
	#rectified linear Unit(Relu)
	#it calculates max(x,0) for each input pixel x
	#this add some non-linearit to the formula and allow us
	#to learn more complicated functions
	layer = tf.nn.relu(layer)
	
	#note that relu is normally executed before pooling,
	#but since relu(max_pool(x))==max_pool(relu(x)) we can
	#save 75% of the relu operations by max-pooling first
	
	#we return both the resulting laer and the filter weights
	#because we will plot the weights later
	return layer, weights
	
	
#helper function for flattening a layer

def flatten_layer(layer):
	#Get the shape of input layer
	layer_shape = layer.get_shape()
	
	#the shape of input layer is assumed to be:
	#layer_shape = [num_images, img_height, img_width, num_channels]
	
	#the number of features is img_height*img_widht*num_channels
	#we can use a function from Tensorflow to calculate this
	num_features = layer_shape[1:4].num_elements()
	
	
	#reshape the layer to []num_images, num_features]
	#note that we just set the size of second dimension
	#to num_features and the size of the first dimension to -1
	#which means the size of the tensor is unchanged from reshaping
	layer_flat = tf.reshape(layer, [-1, num_features])
	
	#shape of flattened layer is now:
	#[num_images, img_height*img_weight*num_channels]
	
	#return both flattened layer and num of features
	return layer_flat, num_features
	

#hhelper function for creating a new fully connected layer
def new_fc_layer(input,
				num_inputs,
				num_outputs,
				use_relu = True)
				
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length = num_outputs)
	
	#calcuate the layer as matrix multiplicatiopn of
	#the input and weights and then add the biasees alues
	layey = tf.matmul(input,weights)+ biases
	
	#use relu
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

	
x = tf.placeholder (tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name = 'y_true')

y_true_cls = tf.argmax(y_true, dimension=1)


#concv layer 1
layer_conv1, weights_conv1 = \
	new_conv_layer(input = x_image,
					num_input_channels = num_channels,
					filter_size= filter_size1,
					num_filters = num_filters1,
					use_pooling=True)
					
layer_con1

#conv layer 2

layer_conv2, weights_convv2 = \
	new_conv_layer(input = layer_conv1,
					num_input_channels = num_filters1,
					filter_size = filter_size2,
					num_filters = num_filters2,
					use_pooling = True)
					
layer_convv2

#flatten layyer

layer_flat, num_features = flatten_layer(layer_conv2)

layer_flat

num_features

#fully_connected layer 2
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc1
						 
#fully_connected layer 2

layer_fc1 = new_fc_layer(input= layer_fc1,
						num_inputs = fc_size,
						num_outputs = num_classes,
						use_relu = False)
						
layer_fc2

#predicted class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)


#cost function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= layer_fc2, 
														labels = y_true)

cost = tf.reduce_mean(cross_entropy)


#optimization method

optimizer  = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

#perfomrance measures

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Create TensorFlow session
session.tf.session()

#Initialize variables
session.run(tf.global_variables_intializer())

#helper function to perform optimization iterations
train_batch_size= 64

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
	#ensure we update the global variable rather than a local copy
	global total_iterations
	
	#start-time used for printing time-usage below
	start_time = time.time()
	
	for i in range(total_iterations,
					total_iterations+num_iterations):
		#get a batch of training examples
		#x_batch now holds a batch of images and 
		#y_true_batch are the true labesl for those images
		x_batch, y_true_batch = data.train.next_batch(train_batch_size)
		
		#put the batch into a dict with the proper names
		#for placeholder variables in the tensorflow graph
		feed_dict_train = {x:x_batch,
							y_true:y_true_batch}
		
		#run the optimizer using the batch of training data
		#tensorflow assigns the variables in feed_dict_train
		#to the placeholder variables and then run the optimizer
		session.run(optimizer, feed_dict = feed_dict_train)
		
	#update the total number of iterations performed
	total_iterations += num_iterations
		
	#ending time
	end_time = time.time()
		
	#difference between start and end-times
	time_dif = end_time - start_time
		
	#print the time-usage
	print("Time usage:"+str(timedelta(seconds=int(round(time_diff)))))

	
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
		

print_test_accuracy()

optimize(num_iterations=1)

print_test_accuracy()

optimize(num_iterations=99) # We already performed 1 iteration above.

print_test_accuracy(show_example_errors=True)

optimize(num_iterations=900) # We performed 100 iterations above.

print_test_accuracy(show_example_errors=True)

optimize(num_iterations=9000) # We performed 1000 iterations above.


print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
					

session.close()
