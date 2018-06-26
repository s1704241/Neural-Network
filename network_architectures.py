import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops.nn_ops import leaky_relu,relu

from utils.network_summary import count_parameters


class VGGClassifier:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=2, strided_dim_reduction=True):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('VGGNet'):
                outputs = image_input
                for i in range(len(self.layer_stage_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                if (j == self.inner_layer_depth-1) and self.strided_dim_reduction:
                                    stride = 2
                                else:
                                    stride = 1
                                outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3],
                                                           strides=(stride, stride),
                                                           padding='SAME', activation=None)
                                outputs = leaky_relu(outputs, name="leaky_relu{}".format(i))
                                layer_features.append(outputs)
                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                         center=True, is_training=training, renorm=False)
                        if self.strided_dim_reduction==False:
                            outputs = tf.layers.max_pooling2d(outputs, pool_size=(2, 2), strides=2)

                        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
                                                                              # apply dropout only at dimensionality
                                                                              # reducing steps, i.e. the last layer in
                                                                              # every group

            c_conv_encoder = outputs
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "VGGNet")

        return c_conv_encoder, layer_features


class FCCLayerClassifier:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=5, strided_dim_reduction=True):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('FCCLayerNet'):
                outputs = image_input
                for i in range(len(self.layer_stage_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                outputs = tf.layers.dense(outputs, units=self.layer_stage_sizes[i])
                                outputs = leaky_relu(outputs, name="leaky_relu{}".format(i))
                                layer_features.append(outputs)
                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                         center=True, is_training=training, renorm=False)
                        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
                                                                              # apply dropout only at dimensionality
                                                                              # reducing steps, i.e. the last layer in
                                                                              # every group

            c_conv_encoder = outputs
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "FCCLayerNet")

        return c_conv_encoder, layer_features

class Baseline:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=8, strided_dim_reduction=True):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        stride=1

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('Baseline'):
                outputs = image_input

                #outputs = tf.layers.dropout(outputs, rate=0.2, training=training)

                outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[0], [3, 3], strides=(stride, stride), padding='SAME', activation=None)
               
                if self.batch_norm_use:
                    outputs = batch_norm(outputs, decay=0.99, scale=True, center=True, is_training=training, renorm=False)

                #outputs = relu(outputs, name="relu")
                outputs =1.125*outputs*tf.nn.sigmoid(outputs, name="swish")

                #outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[0], [3, 3], strides=(stride, stride), padding='SAME', activation=None, kernel_regularizer=regularizer)  

                for i in range(len(self.layer_stage_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                if j==0 and i!=0:
                                    stride=2
                                    res=tf.layers.average_pooling2d(outputs, pool_size=(2, 2), strides=2)
                                    res=tf.concat([res,tf.zeros_like(res)],axis=3)
                                   
                                else:
                                    stride=1
                                    res=outputs
                                   

                                outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3], strides=(stride, stride), padding='SAME', activation=None)
                                #outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3], strides=(stride, stride), padding='SAME', activation=None, kernel_regularizer=regularizer)



                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True, center=True, is_training=training, renorm=False)

                               # outputs = relu(outputs, name="relu{}".format(i))
                               # outputs = tf.layers.dropout(outputs, rate=0.5, training=training)   
                                outputs =1.125*outputs*tf.nn.sigmoid(outputs, name="swish{}".format(i))

                               # outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3], strides=(stride, stride), padding='SAME', activation=None, kernel_regularizer=regularizer)

                                layer_features.append(outputs)

                                stride=1

                                outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3], strides=(stride, stride), padding='SAME', activation=None)
                                #outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3], strides=(stride, stride), padding='SAME', activation=None, kernel_regularizer=regularizer)



                                #outputs =outputs*tf.nn.sigmoid(outputs, name="swish{}".format(i))

                               # outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3], strides=(stride, stride), padding='SAME', activation=None, kernel_regularizer=regularizer)

                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True, center=True, is_training=training, renorm=False)

                         
                                outputs=outputs+res

                                #outputs = relu(outputs, name="relu{}".format(i))
                                outputs =1.125*outputs*tf.nn.sigmoid(outputs, name="swish{}".format(i))


                                layer_features.append(outputs)

                        
                        #outputs = tf.layers.dropout(outputs, rate=0.5, training=training)
                        



            #outputs = tf.layers.average_pooling2d(outputs, pool_size=(2, 2), strides=1)

            c_conv_encoder = outputs
            c_conv_encoder = tf.reduce_mean(c_conv_encoder,[1,2])
           # c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "Baseline")

        return c_conv_encoder, layer_features


class Pyramid:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                inner_layer_depth=8, strided_dim_reduction=True):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes=layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        self.total_layer_depth=inner_layer_depth*3
        alpha=64*inner_layer_depth
        for i in range(1,self.total_layer_depth+1):
            self.layer_stage_sizes.append((layer_stage_sizes[0]+(alpha*i)//self.total_layer_depth))
            

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        stride=1
        self.former_dim=self.layer_stage_sizes[0]
        self.current_dim=self.layer_stage_sizes[0]

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('Pyramid'):
                outputs = image_input

                #outputs = tf.layers.dropout(outputs, rate=0.2, training=training)

                outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[0], [3, 3], strides=(stride, stride), padding='SAME', activation=None)
               
                if self.batch_norm_use:
                    outputs = batch_norm(outputs, decay=0.99, scale=True, center=True, is_training=training, renorm=False)

                #outputs = relu(outputs, name="relu")
                outputs = 1.125*outputs*tf.nn.sigmoid(outputs, name="swish")



                #outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[0], [3, 3], strides=(stride, stride), padding='SAME', activation=None, kernel_regularizer=regularizer)

                for i in range(3):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                if j==0 and i!=0:
                                    stride=2
                                    res=tf.layers.average_pooling2d(outputs, pool_size=(2, 2), strides=2)
                                   
                                else:
                                    stride=1
                                    res=outputs

                                self.former_dim=self.current_dim
                                self.current_dim=self.layer_stage_sizes[i*self.inner_layer_depth+j]

                                res=tf.concat([res,tf.zeros([res.shape[0],res.shape[1],res.shape[2],self.current_dim-self.former_dim])],axis=3)                                   

                                outputs = tf.layers.conv2d(outputs, self.current_dim, [3, 3], strides=(stride, stride), padding='SAME', activation=None)


                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True, center=True, is_training=training, renorm=False)

                                #outputs = relu(outputs, name="relu{}".format(i))

                                outputs =1.125*outputs*tf.nn.sigmoid(outputs, name="swish{}".format(i))

                               # outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3], strides=(stride, stride), padding='SAME', activation=None, kernel_regularizer=regularizer)

                                layer_features.append(outputs)

                                stride=1

                                outputs = tf.layers.conv2d(outputs, self.current_dim, [3, 3], strides=(stride, stride), padding='SAME', activation=None)


                                #outputs =outputs*tf.nn.sigmoid(outputs, name="swish{}".format(i))

                               # outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], [3, 3], strides=(stride, stride), padding='SAME', activation=None, kernel_regularizer=regularizer)

                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True, center=True, is_training=training, renorm=False)

                         
                                outputs=outputs+res

                                #outputs = relu(outputs, name="relu{}".format(i))
                                outputs =1.125*outputs*tf.nn.sigmoid(outputs, name="swish{}".format(i))



                                layer_features.append(outputs)


                        #outputs = tf.layers.dropout(outputs, rate=0.5, training=training)



            #outputs = tf.layers.average_pooling2d(outputs, pool_size=(2, 2), strides=1)

            c_conv_encoder = outputs
            c_conv_encoder = tf.reduce_mean(c_conv_encoder,[1,2])
           # c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "Pyramid")

        return c_conv_encoder, layer_features

