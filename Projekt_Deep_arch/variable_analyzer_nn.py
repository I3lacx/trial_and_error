
# coding: utf-8

# In[1]:


from observations import mnist
import numpy as np
import tensorflow as tf
import os


# In[7]:


def load_mnist():
    """ loads data from mnist function and converts to numpy arrays"""

    (train_im, train_lab), (test_im, test_lab) = mnist('data/mnist')
    train_im_mean = np.mean(train_im, 0)
    train_im_std = np.std(train_im, 0)

    std_eps = 1e-7
    train_im = np.array((train_im - train_im_mean) / (train_im_std + std_eps))
    test_im = np.array((test_im - train_im_mean) / (train_im_std + std_eps))

    #convert to dictionary
    data = {}
    data['x_train'] = train_im
    data['y_train'] = np.array(train_lab)
    data['x_test'] = test_im
    data['y_test'] = np.array(test_lab)

    return data

def get_logs_path(session):
    """ Returns path where log is going to be saved.
    @Input session is the parent folder for the runs
    So e.g. (Session1 -> run_00, run_01, run_02 ),
    (Session2 -> run_00, run_01)"""
    curr_dir = os.getcwd()
    logs_path = curr_dir + '/tensorflow_logs/train_mnist/' + session
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    previous_runs = os.listdir(logs_path)
    new_path = logs_path + "/run_{:02d}".format(len(previous_runs))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        print("Saving log at: ", new_path)
        return new_path
    raise Exception("Path is already existing: ", new_path)


# In[3]:

def var_summaries(var):
    """Add summaries of var to tensorboard"""
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def create_weights(pref, batch_size, in_size, out_size):
#TODO: check if this works
    with tf.name_scope("weights"):
        weights = None
        if pref["mode"] == "norm":
            weights = tf.Variable(tf.random_normal((batch_size, out_size, in_size), mean=pref["mean"], stddev=pref["stddev"]))
        elif pref["mode"] == "uniform":
            weights = tf.Variable(tf.random_uniform((batch_size, out_size, in_size), minval=pref["min"], maxval=["max"]))
        elif pref["mode"] == "const":
            weights = tf.Variable(zeros((batch_size, out_size, in_size), dtype=tf.float32) + pref["value"])
        else:
            raise Exception("wrong parameter for mode: {}".format(pref["mode"]))
    return weights


def create_bias(pref, out_size):
    """creates biases with preferences"""
    with tf.name_scope("bias"):
        bias = None
        if pref["mode"] == "norm":
            bias = tf.Variable(tf.random_normal((out_size,), mean=pref["mean"], stddev=pref["stddev"]))
        elif pref["mode"] == "uniform":
            bias = tf.Variable(tf.random_uniform((out_size,), minval=pref["min"], maxval=["max"]))
        elif pref["mode"] == "const":
            bias = tf.Variable(zeros((out_size,), dtype=tf.float32) + pref["value"])
        else:
            raise Exception("wrong parameter for mode: {}".format(pref["mode"]))

    return bias


def dense_layer(inputs, weight_pref, bias_pref, output_size, name):
    """ creates dense layer with given weight and bias preferences"""
    with tf.name_scope(name):
        inputs_shape = inputs.get_shape().as_list()
        weights = create_weights(weight_pref, inputs_shape[0], inputs_shape[1], output_size)
        bias = create_bias(bias_pref, output_size)

        with tf.name_scope("weights"):
            var_summaries(weights)
        with tf.name_scope("bias"):
            var_summaries(bias)

        #expand dimension of inputs for matmul
        inputs = tf.expand_dims(inputs, -1)
        out = tf.add(tf.squeeze(tf.matmul(weights, inputs)), bias)
    return out


""" Ziel ist eine nn struktur die sich durch tweaken von eins zwei parametern komplett verändern lässt
dabei sollte die depth variable sein, so wie wie viele hidden units ich habe
wie die variablen aufgesetzt werden sollte auch per input regulierbar sein
Maybe with dropout etc."""
def forward(input_samples, num_hidden=2, output_units=[20,10,10,10]):
    """ this is the whole network structure, returns the last layer after computing
    every other layer"""
    
    #assert that inputs are vaguely corretct:
    assert len(output_units) == num_hidden + 2,    "output_units length: {}, while only {} num_hidden".format(len(output_units),  num_hidden)
    
    #reshape input data
    input_layer = input_samples

    pref = {
        "mode": "norm",
        "mean": 0.2,
        "stddev": 1
    }
    weight_pref = pref
    bias_pref = pref

    first_layer = dense_layer(input_layer, weight_pref, bias_pref, output_units[0], name="Input_layer")

    layers = {
        "l0": first_layer 
    }
    
    # some deep layers num_hidden iterations, just shifted one forward
    for i in range(1,num_hidden+1):
        prev_layer = layers["l"+str(i-1)]
        layers["l"+str(i)] = dense_layer(prev_layer, weight_pref, bias_pref,
                                        output_units[i], name="Hidden_layer_"+str(i))
    
    out_layer = dense_layer(layers["l"+str(num_hidden)], weight_pref, bias_pref, output_units[-1], name="Output_Layer")
    return out_layer

# In[4]:


def train(x_train, y_train, batch_size, input_size, num_epochs, writer):
    """ trains the network defined in forward """
    
    x_ph = tf.placeholder(tf.float32, [batch_size, input_size], name="x_ph")
    y_ph = tf.placeholder(tf.int32, [batch_size], name="y_ph")

    prediction = forward(x_ph)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_ph, logits=prediction)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    
    tf.summary.scalar("loss",loss)
    summ = tf.summary.merge_all()
    
    batches_per_epoch = x_train.shape[0] // batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(num_epochs):
            for batch in range(batches_per_epoch):
                x_batch = x_train[batch * batch_size : (batch+1) * batch_size]
                y_batch = y_train[batch * batch_size : (batch+1) * batch_size]
                
                _, cur_loss, cur_summ = sess.run([train_op, loss, summ], feed_dict={x_ph: x_batch, y_ph: y_batch})
                writer.add_summary(cur_summ, epoch*batches_per_epoch + batch)
                #print(cur_loss)


# In[5]:


def run_model():
    mnist_data = load_mnist()
    logs_path = get_logs_path("Session_00")
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    train(x_train=mnist_data["x_train"], y_train=mnist_data["y_train"],
     batch_size=100, input_size=784, num_epochs=5, writer=writer)


# In[8]:


run_model()

pref = {
    "mode": "norm",
    "mean": 0.2,
    "stddev": 1
}

weights = create_weights(pref, 100, 784, 20)

print(weights)