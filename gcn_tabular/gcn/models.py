from gcn.layers import *
from gcn.metrics import *
from gcn.utils import *
import gcn.globs as g
import pdb

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        # for kwarg in kwargs.keys():
        #     assert kwarg in allowed_kwargs, 'Invbalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss,global_step=self.global_step)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, edges, laplacian, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)
        # pdb.set_trace()
        self.edges=edges
        self.laplacian=laplacian
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.global_step = tf.Variable(0, trainable=False)

        # self.learning_rate = tf.train.exponential_decay(0.01, self.global_step,
        #                                            FLAGS.epochs, 0.96, staircase=True)
        self.learning_rate = placeholders['learning_rate']
        # pdb.set_trace()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate )



        self.build()

        # self.lols = tf.slice(self.outputs,[0,0],[5,0])
    def _loss(self):
        # Weight decay loss
        # pdb.set_trace()
        for layer in self.layers:
            for var in layer.vars.values():
                if "bias" in var.name:
                    continue
                # pdb.set_trace()
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # sess = tf.Session();sess.run(tf.global_variables_initializer());pdb.set_trace()
        """
        .eval(session=sess,feed_dict=g.feedz)
        """   

        outs = tf.nn.softmax(self.outputs)[:,1]

        all_probs_outs=tf.nn.softmax(self.outputs)
        self.entropy =  - tf.reduce_sum(all_probs_outs * tf.log(all_probs_outs))
        # self.loss += -1e-2*self.entropy

        first = tf.matmul(tf.transpose(tf.expand_dims(outs,1)),self.laplacian)
        total = tf.matmul(first,tf.expand_dims(outs,1))
        self.midloss = tf.reduce_sum(total) / len(self.edges)
        # pdb.set_trace()
        # self.loss += 1*10**(-int(FLAGS.fig))*self.midloss
        self.loss += 3e-0*self.midloss



        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

        # self.loss += masked_l2loss(self.outputs, self.placeholders['labels'],
        #                                           self.placeholders['labels_mask'])        

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        act = tf.nn.relu
        hid1 = FLAGS.hidden1
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=act,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
        #                                     output_dim=FLAGS.hidden2,
        #                                     placeholders=self.placeholders,
        #                                     act=act,
        #                                     dropout=True,
        #                                     # sparse_inputs=True,
        #                                     logging=self.logging))
        # FLAGS.hidden1= FLAGS.hidden2


        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
        #                                     output_dim=FLAGS.hidden3,
        #                                     placeholders=self.placeholders,
        #                                     act=act,
        #                                     dropout=True,
        #                                     # sparse_inputs=True,
        #                                     logging=self.logging))
        # FLAGS.hidden1= FLAGS.hidden3

        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden3,
        #                                     output_dim=FLAGS.hidden4,
        #                                     placeholders=self.placeholders,
        #                                     act=act,
        #                                     dropout=True,
        #                                     # sparse_inputs=True,
        #                                     logging=self.logging)) 
        # FLAGS.hidden1= FLAGS.hidden4

        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden4,
        #                                     output_dim=FLAGS.hidden5,
        #                                     placeholders=self.placeholders,
        #                                     act=act,
        #                                     dropout=True,
        #                                     # sparse_inputs=True,
        #                                     logging=self.logging)) 
        # FLAGS.hidden1= FLAGS.hidden5        
                                                                                               
        
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            # act=tf.nn.softmax,
                                            dropout=True,
                                            logging=self.logging))
        FLAGS.hidden1 = hid1

    def predict(self):
        return tf.nn.softmax(self.outputs)




class MLP(Model):
    def __init__(self, placeholders, edges,laplacian, input_dim, **kwargs):
        # pdb.set_trace()
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

