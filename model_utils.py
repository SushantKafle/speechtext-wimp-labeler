import tensorflow as tf

def get_rnn_cell(size, type_, state_is_tuple = True):
	if type_ == "LSTM":
		return tf.contrib.rnn.LSTMCell(size, state_is_tuple=state_is_tuple)
	elif type_ == "GRU":
		return tf.contrib.rnn.GRUCell(size)

def create_feedforward(input_tensor, input_size, output_size, fn_initializer, activation, scope):
	with tf.variable_scope(scope):
		weights = tf.get_variable("W_", dtype = tf.float32, initializer = fn_initializer((input_size, output_size)))
		bias = tf.get_variable("b_", dtype = tf.float32, initializer = fn_initializer((output_size,)))
		output = tf.matmul(input_tensor, weights) + bias
		if activation == "tanh":
			output = tf.tanh(output)
		elif activation == "sigmoid":
			output = tf.sigmoid(output)
		return output