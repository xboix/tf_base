"""
The abstract model
"""
import tensorflow as tf
import numpy as np

class Network(tf.keras.Model):

    def __init__(self, num_classes):
        super(Network, self).__init__()

        self.num_classes = num_classes
        self.train_variables = []
        pass

    def feedforward_pass(self, input):
        pass

    def __call__(self, input):
        self.x_input = input
        return self.feedforward_pass(input)

    @tf.function
    def train_step(self, input, label):
        self._full_call(input, label, evaluate=False)

    @tf.function
    def evaluate(self, input, label,  step=-1, summary=None):
        return self._full_call(input, label, step=step,
                        evaluate=True, summary=summary)


    def _full_call(self, input, label,  evaluate=False,  summary=None, step=-1):

        self.x_input = input
        self.y_input = label

        with tf.GradientTape() as self.tape:

            self.feedforward_pass(self.x_input)

            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=tf.cast(self.y_input, tf.int32), logits=self.pre_softmax)
            self.loss = tf.reduce_mean(y_xent)


        if not evaluate:
            self.optimizer.apply_gradients(zip(self.tape.gradient(self.loss, self.train_variables),
                                               self.train_variables))
        else:
            # Evaluation
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(label, tf.int32), logits=self.pre_softmax)
            self.xent = tf.reduce_mean(y_xent)

            self.y_pred = tf.argmax(self.pre_softmax, 1)
            correct_prediction = tf.equal(self.y_pred, label)
            self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return self.accuracy

            if summary:
                with summary.as_default():
                    tf.summary.scalar('Cross Entropy', self.xent, step)
                    tf.summary.scalar('Accuracy', self.accuracy, step)
                    tf.summary.scalar('Learning Rate', self.optimizer.learning_rate(step), step)


    def load_all(self, path, load_optimizer=True):

        if load_optimizer:
            opt_weights = np.load(path + '_optimizer.npy', allow_pickle=True)

            grad_vars = self.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
            self.optimizer.set_weights(opt_weights)

        self.load_weights(path)

    def save_all(self, path):
        self.save_weights(path)
        np.save(path + '_optimizer.npy', self.optimizer.get_weights())


def get_network(name_net, config, num_features):
    if name_net == 'CNN':
        from networks.CNN import CNN
        return CNN(config, num_features)

