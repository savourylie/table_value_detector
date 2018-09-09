import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# Hyperparams:
g_keep_prob = 0.7
g_learning_rate = 1e-4
g_batch_size = 16

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    vgg_model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3, layer4, layer7
    
tests.test_load_vgg(load_vgg, tf)


def conv_1x1(layer, num_classes, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), layer_name='default_layer'):
    return tf.layers.conv2d(layer, num_classes, 1, 1, padding=padding, kernel_regularizer=kernel_regularizer, name=layer_name)


def upsample(layer, num_outputs, kernel, strides, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), layer_name='default'):
    """
    Apply a two times upsample on x and return the result.
    :x: 4-Rank Tensor
    :return: TF Operation
    """

    return tf.layers.conv2d_transpose(layer, num_outputs, kernel, strides, padding=padding, kernel_regularizer=kernel_regularizer, name=layer_name)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Apply a 1x1 convolution to encoder layers
    l7_conv1x1 = conv_1x1(vgg_layer7_out, num_classes, layer_name="l7_conv1x1")
    layer4_skip = conv_1x1(vgg_layer4_out, num_classes, layer_name="l4_conv1x1")
    layer3_skip = conv_1x1(vgg_layer3_out, num_classes, layer_name="l3_conv1x1")

    # Add decoder layers to the network with skip connections and upsampling
    # Note: the kernel size and strides are the same as the example in Udacity Lectures
    #       Semantic Segmentation Scene Understanding Lesson 10-9: FCN-8 - Decoder
    decoderlayer1 = upsample(layer=l7_conv1x1, num_outputs=num_classes, kernel=4, strides=2, layer_name="decoderlayer1")
    decoderlayer2 = tf.add(decoderlayer1, layer4_skip, name="decoderlayer2")
    decoderlayer3 = upsample(layer=decoderlayer2, num_outputs=num_classes, kernel=4, strides=2, layer_name="decoderlayer3")
    decoderlayer4 = tf.add(decoderlayer3, layer3_skip, name="decoderlayer4")

    decoderlayer_output = upsample(layer=decoderlayer4, num_outputs=num_classes, kernel=16, strides=8, layer_name="decoderlayer_output")

    return decoderlayer_output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    for epoch in range(epochs):
        batch_loss = []

        for image, label in get_batches_fn(batch_size):
            feed_dict = {input_image: image, correct_label: label, keep_prob: g_keep_prob, learning_rate: g_learning_rate}

            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            batch_loss.append(loss)

        print("Epoch {0}: Training loss: {1}".format(epoch + 1, sum(batch_loss)))


tests.test_train_nn(train_nn)


def run():
    start_time = time.time()

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Placeholder tensors
    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)


    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        decoderlayer_output = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(decoderlayer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())

        train_nn(sess, 160, g_batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


        # OPTIONAL: Apply the trained model to a video

    print("Training and predicting took {} seconds.".format(time.time() - start_time))

if __name__ == '__main__':
    run()
