"""
Text To Speech (TTS) Neural Network
"""

# Imports
import numpy as np
import soundfile as sf
import tensorflow as tf

# Definitions
ascii_size = 256
clarity = 10  # clarity of sound
epochs = 100
layers = 1  # number of layers
learn_rate = 1e-1  # learning rate
neurons = 10  # number of neurons in layers
predict_filename = "predict"  # output filename
progress_every = 1
sound_ext = ".wav" # extension for sound file type
tf_type = tf.float32  # value type in tensorflow
time_step = 1  # set time step

# Get sound and text file
sound_filename = input("Enter the (.wav) file to train: ")  # get sound file
sound, samp_rate = sf.read(sound_filename)  # open sound file
try:
    streams = len(sound[0])  # get number of streams
except Exception:
    streams = 1  # only one stream
text_filename = input("Enter the (.txt) File to train: ")  # get text file
with open(text_filename, "r") as f:  # open text
    text = f.read()  # read text file
len_text = len(text)  # get text length
samp_per_char = int(len(sound) / len_text)  # get number of samples per character
sound = sound[:samp_per_char * len_text]
len_sound = len(sound)  # get sound length
batch_size = int(len_sound / samp_per_char)  # get number of time steps
print("Sound length(s): {0}, text characters: {1}, samples per character: {2}, Time Steps: {3}".format(
                                                                                      len_sound / samp_rate,
                                                                                      len_text,
                                                                                      samp_per_char,
                                                                                      time_step))

# Create network
net_input = tf.placeholder(tf_type, (batch_size, time_step, ascii_size), name="input")  # input to network
labels = tf.placeholder(tf_type, (batch_size, time_step, samp_per_char, streams,
                                  clarity), name="labels")  # labels to train with
cell_layer = tf.nn.rnn_cell.LSTMCell(neurons)
cell = tf.nn.rnn_cell.MultiRNNCell([cell_layer] * layers)  # create network

zero_state = cell.zero_state(batch_size, tf_type)
lstm_output, new_state = tf.nn.dynamic_rnn(cell, net_input, initial_state=zero_state)  # run lstm
ff_out = tf.layers.dense(lstm_output, samp_per_char * streams * clarity, name="logits")  # create result
logits = tf.reshape(ff_out, (batch_size, time_step, samp_per_char, streams, clarity), name="logits")  # reshape

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
                      axis=0, name="cost")  # get cost. Average over batch
mean_cost = tf.reduce_mean(cost, name="mean_cost")  # get mean cost
optim = tf.train.RMSPropOptimizer(learn_rate).minimize(cost)  # create optimizer

pred = tf.argmax(logits, axis=-1, name="pred")  # get prediction
n_correct = tf.equal(tf.argmax(labels, axis=-1), pred, name="n_correct")
accur = tf.reduce_mean(tf.cast(n_correct, tf_type), name="accur")  # get accuracy
init_op = tf.global_variables_initializer()  # create initializer
sess = tf.Session()  # create session
sess.run([init_op])  # run initialization

# Train
train_input = np.zeros((batch_size, time_step, ascii_size))  # create input
for i_char in range(len_text):
    batch = int(i_char / time_step)
    step = i_char % time_step  # get step
    train_input[batch][step][ord(text[i_char])] = 1.0  # set input value

train_label = np.zeros((batch_size, time_step, samp_per_char, streams, clarity))  # create labels
for i_sound in range(len_sound):
    batch = int(i_sound / (time_step * samp_per_char))  # get batch
    step = int(i_sound % (time_step * samp_per_char) / samp_per_char)  # get current step
    sample = i_sound % samp_per_char  # get current sample
    if streams > 1:  # if more than one stream
        for i_stream in range(streams):  # cycle through streams
            train_label[batch][step][sample][i_stream][int((sound[i_sound][i_stream] + 1) / 2 * clarity)] = 1.0
    else:  # if mono
        train_label[batch][step][sample][0][int((sound[i_sound] + 1) / 2 * clarity)] = 1.0

for i_epoch in range(epochs):
    if i_epoch % progress_every == 0:  # if write progress
        _, epoch_cost, epoch_accur = sess.run([optim, mean_cost, accur],
                                             {net_input: train_input,
                                              labels: train_label})  # train
        print("Epoch {0}: Accuracy {1}, Cost {2}".format(i_epoch, epoch_accur, epoch_cost))  # write progress
    else:
        sess.run(optim, {net_input: train_input,
                         labels: train_label})  # train

# Predict
is_exit = False  # should exit
predict_index = 0
while not is_exit:  # while running
    predict_text = input("Enter text to speak (Type 'Exit' to close)(max {0} characters: ".
                         format(time_step))  # get text
    if predict_text.lower() == "exit":
        is_exit = True
    else:  # else predict
        predict_input = np.zeros((batch_size, time_step, ascii_size))  # create input
        for i_char in range(len(predict_text)):  # cycle through text
            batch = int(i_char / time_step)
            step = i_char % time_step
            predict_input[batch][step][ord(predict_text[i_char])] = 1.0  # get input

        predict_result = sess.run([pred], {net_input: predict_input})  # run session
        if streams > 1:
            predict_result = np.reshape(predict_result, (-1, streams))  # reshape
        else:
            predict_result = np.reshape(predict_result, (-1))  # reshape
        predict_result = predict_result[:(len(predict_text) * samp_per_char)]  # slice to size
        predict_result = (predict_result * 2 / clarity) - 1  # scale
        sf.write(predict_filename + str(predict_index) + sound_ext,
                 predict_result, samp_rate)  # write result
        predict_index += 1  # increment index
