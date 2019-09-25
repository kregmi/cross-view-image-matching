import tensorflow as tf


class Siamese_FC:

    def fc_layer(self, x, input_dim, output_dim, init_dev, init_bias,
                 trainable, name='fc_layer', activation_fn=tf.nn.relu):
        with tf.variable_scope(name):
            print (name)
            
            print (x)
            weight = tf.get_variable(name='weights', shape=[input_dim, output_dim],
                                     trainable=trainable,
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=init_dev))
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.constant_initializer(init_bias))

            if activation_fn is not None:
                out = tf.nn.xw_plus_b(x, weight, bias)
                out = activation_fn(out)
            else:
                out = tf.nn.xw_plus_b(x, weight, bias)

        return out



    def three_stream_fc(self, x_sat, x_grd, x_grd_seg, trainable, scope_name):
        print('Siamese_FC:', scope_name, ' trainable =', trainable)

        with tf.variable_scope(scope_name) as scope:
            fc_sat = self.fc_layer(x_sat, 4*4*512 + 8*8*512 + 4*8*8*512, 1000, 0.005, 0.1, trainable, 'fc1', activation_fn=None)
            sat_global = tf.nn.l2_normalize(fc_sat, dim=1)


            fc_grd = self.fc_layer(x_grd, 190976, 1000, 0.005, 0.1, trainable, 'fc2', activation_fn=None)
            grd_global_0 = tf.nn.l2_normalize(fc_grd, dim=1)
            
            scope.reuse_variables()

            fc_sat_synth = self.fc_layer(x_grd_seg, 4*4*512 + 8*8*512 + 4*8*8*512, 1000, 0.005, 0.1, trainable, 'fc1', activation_fn=None)
            grd_global_1 = tf.nn.l2_normalize(fc_sat_synth, dim=1)

  #          scope.reuse_variables()

            concat_grd = tf.concat([grd_global_0, grd_global_1], 1)

            fc_grd_global = tf.contrib.layers.fully_connected( concat_grd,1000, scope='fc2_after_concat',                                            activation_fn=None, reuse=tf.AUTO_REUSE) 

            grd_global = tf.nn.l2_normalize(fc_grd_global, dim=1)

        return sat_global, grd_global




    def three_stream_joint_feat_learning(self, x_sat, x_grd, x_grd_gan, trainable, scope_name):
        print('Siamese_FC:', scope_name, ' trainable =', trainable)

        with tf.variable_scope(scope_name) as scope:
            fc_grd = self.fc_layer(x_grd, 53760, 1000, 0.005, 0.1, trainable, 'fc2', activation_fn=None)
            grd_global = tf.nn.l2_normalize(fc_grd, dim=1)


            fc_sat = self.fc_layer(x_sat, 43008, 1000, 0.005, 0.1, trainable,
                                   'fc1', activation_fn=None)
            sat_global = tf.nn.l2_normalize(fc_sat, dim=1)


            scope.reuse_variables()


            fc_sat_synth = self.fc_layer(x_grd_gan, 43008, 1000, 0.005, 0.1,
                                         trainable, 'fc1', activation_fn=None)
            grd_sat_gan = tf.nn.l2_normalize(fc_sat_synth, dim=1)

        return sat_global, grd_global, grd_sat_gan

