import tensorflow as tf


class VGG16:

    ############################ kernels #############################
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1],
                            padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    ############################ layers ###############################
    def conv_layer(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            if activated:
                out = activation_function(self.conv2d(x, weight) + bias)
            else:
                out = self.conv2d(x, weight) + bias

            return out

    def maxpool_layer(self, x, name):
        with tf.name_scope(name):
            maxpool = self.max_pool_2x2(x)
            return maxpool


    def eight_layer_conv(self, x, keep_prob, trainable, name):
         print('eight_layer_network: trainable = ', trainable)

         with tf.variable_scope(name):

            # layer 1: conv 3-64
            layer1_output = self.conv_layer(x, 4, 3, 64, trainable, True, 'conv1')

            # layer 2: conv 64 - 256
            layer2_output = self.conv_layer(layer1_output, 4, 64, 128, trainable, True, 'conv2')

            # layer 3: conv 256 - 512
            layer3_output = self.conv_layer(layer2_output, 4, 128, 256, trainable, True, 'conv3')

            # layer 4: conv 512 - 512
            layer4_output = self.conv_layer(layer3_output, 4, 256, 512, trainable, True, 'conv4')

            # layer 5: conv 512 - 512
            layer5_output = self.conv_layer(layer4_output, 4, 512, 512, trainable, True, 'conv5')

            # layer 6: conv 512 - 512
            layer6_output = self.conv_layer(layer5_output, 4, 512, 512, trainable, True, 'conv6')
            layer6_output = tf.nn.dropout(layer6_output, keep_prob, name='conv6_dropout')

            # layer 7: conv 512 - 512
            layer7_output = self.conv_layer(layer6_output, 3, 512, 512, trainable, True, 'conv7')
            layer7_output = tf.nn.dropout(layer7_output, keep_prob, name='conv7_dropout')


            # layer 8: conv 512 - 512
            layer8_output = self.conv_layer(layer7_output, 4, 512, 512, trainable, True, 'conv8')
            layer8_output = tf.nn.dropout(layer8_output, keep_prob, name='conv8_dropout')
            layer8_b_output = tf.layers.flatten( layer8_output, 'reshape_feats_8')


            return layer8_output



    def eight_layer_conv_multiscale(self, x, keep_prob, trainable, name):
        print('eight_layer_network_multiscale: trainable = ', trainable)

        with tf.variable_scope(name):

            # layer 1: conv 3-64
            layer1_output = self.conv_layer(x, 4, 3, 64, trainable, True, 'conv1')

            # layer 2: conv 64 - 256
            layer2_output = self.conv_layer(layer1_output, 4, 64, 128, trainable, True, 'conv2')

            # layer 3: conv 256 - 512
            layer3_output = self.conv_layer(layer2_output, 4, 128, 256, trainable, True, 'conv3')

            # layer 4: conv 512 - 512
            layer4_output = self.conv_layer(layer3_output, 4, 256, 512, trainable, True, 'conv4')

            # layer 5: conv 512 - 512
            layer5_output = self.conv_layer(layer4_output, 4, 512, 512, trainable, True, 'conv5')

            # layer 6: conv 512 - 512
            layer6_output = self.conv_layer(layer5_output, 4, 512, 512, trainable, True, 'conv6')
            layer6_output = tf.nn.dropout(layer6_output, keep_prob, name='conv6_dropout')
            layer6_b_output = tf.layers.flatten( layer6_output, 'reshape_feats_6')


            # layer 7: conv 512 - 512
            layer7_output = self.conv_layer(layer6_output, 4, 512, 512, trainable, True, 'conv7')
            layer7_output = tf.nn.dropout(layer7_output, keep_prob, name='conv7_dropout')
            layer7_b_output = tf.layers.flatten( layer7_output, 'reshape_feats_7')


            # layer 8: conv 512 - 512
            layer8_output = self.conv_layer(layer7_output, 4, 512, 512, trainable, True, 'conv8')
            layer8_output = tf.nn.dropout(layer8_output, keep_prob, name='conv8_dropout')
            layer8_b_output = tf.layers.flatten( layer8_output, 'reshape_feats_8')


            layer9_output = tf.concat([layer6_b_output, layer7_b_output, layer8_b_output], 1)

            return layer9_output


