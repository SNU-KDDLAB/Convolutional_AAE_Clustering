import os, math, sys
data_dir = 'Your_data_path'
out_dir = 'Your_output_path'
change_LR_epoch = -1
load = False
use_CNN = True
turn_on_batch_norm = True
epoch_num = 15
training_keep_prob = 0.8
batch_norm_decay = 0.999
batch_norm_scale = True
NLL = -1
use_adam = False
square = False
fast_img_save = False
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import tensorflow as tf
from myLib import Timer

input_dim = 784
dim1 = 3000
label_dim = 30
z_dim = 5
eps = 1e-8

summary_dir = os.path.join(out_dir, 'summary')
def maybe_make_dir(path):
  if not os.path.exists(path): os.makedirs(path)
maybe_make_dir(out_dir)
maybe_make_dir(summary_dir)

sys.stdout = open(os.path.join(out_dir, 'log.txt'), 'a', 1)
np.set_printoptions(precision=2, threshold=np.nan, linewidth=np.nan)
mb_size = 100

def nn_layer(name, input_data, output_size, reuse = 0):
  if reuse : 
       with tf.variable_scope(name, reuse = True) as scope:
          scope.reuse_variables()
          W = tf.get_variable(name = name + '_W',
                              shape = [input_data.get_shape().as_list()[1], output_size])
          B = tf.get_variable(name = name + '_B',
                              shape = [output_size])
          return tf.matmul(input_data, W) + B
  else:
      with tf.variable_scope(name):
          W = tf.get_variable(name = name + '_W',
                              shape = [input_data.get_shape().as_list()[1], output_size])
          B = tf.get_variable(name = name + '_B',
                              shape = [output_size],
                              initializer = tf.zeros_initializer())
          return tf.matmul(input_data, W) + B

'''plot 16 samples'''
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


'''plot only 1 sample'''    
def plot2(samples):
    fig = plt.figure(figsize=(1, 1))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.05, hspace=0.05)

    #for i, sample in enumerate(samples):
    ax = plt.subplot(gs[0])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(samples.reshape(28, 28), cmap='Greys_r')

    return fig



'''generate gaussian'''
cov = np.identity(z_dim)
def sample_Z(m):
  mu = np.zeros(z_dim)
  return np.random.multivariate_normal(mu, cov , m)
print(sample_Z(3))
'''generate categorial'''
def sample_Y(m):
    return np.eye(label_dim)[np.random.choice(label_dim, m)]



'''''''''''''placeholders'''''''''''''
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, label_dim])
CY = tf.placeholder(tf.float32, shape=[None, label_dim])
Z = tf.placeholder(tf.float32, shape=[None, z_dim])
YZ = tf.placeholder(tf.float32, shape = [None, label_dim + z_dim])
lr1 = tf.placeholder(tf.float32)
lr2 = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool, name='phase')

global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.zeros_initializer(), trainable=False)

input_dropout = tf.nn.dropout(X, keep_prob)
''' Encoder '''
with tf.variable_scope('enc'):
  if use_CNN == False:
    L1 = nn_layer('L1', input_dropout, dim1)
    if turn_on_batch_norm == True:
      L1  = tf.contrib.layers.batch_norm(L1, scale=batch_norm_scale,  is_training=phase, decay=batch_norm_decay)
    L1 = tf.nn.relu(L1)
    L2 = nn_layer('L2', L1, dim1)
    if turn_on_batch_norm == True:
      L2  = tf.contrib.layers.batch_norm(L2, scale=batch_norm_scale,  is_training=phase, decay=batch_norm_decay)
    L2 = tf.nn.relu(L2)
  else: # use CNN
    input_dropout = tf.reshape(input_dropout, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_dropout, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    L2 = tf.layers.dense(inputs=pool2_flat, units=dim1, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'enc')

with tf.variable_scope('gen'):
  z = nn_layer('z', L2, z_dim)
  if turn_on_batch_norm == True:
    z = tf.contrib.layers.batch_norm(z, scale=batch_norm_scale,  is_training=phase, decay=batch_norm_decay)
  y = nn_layer('y', L2, label_dim)
  if turn_on_batch_norm == True:
    y  = tf.contrib.layers.batch_norm(y, scale=batch_norm_scale,  is_training=phase, decay=batch_norm_decay)
  y = tf.nn.softmax(y)
gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')

with tf.variable_scope('dec'):
  L3 = tf.nn.relu(nn_layer('L3', tf.concat([y, z],1), dim1))
  L4 = tf.nn.relu(nn_layer('L4', L3, dim1))
  output = tf.nn.sigmoid(nn_layer('output', L4, input_dim))
dec_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dec')

'''classifier'''
# CL1 = tf.nn.relu(nn_layer('L1', input_dropout, dim1, reuse = 1))
# CL2 = tf.nn.relu(nn_layer('L2', L1, dim1, reuse = 1))
# Cy = tf.nn.softmax(nn_layer('y', L2, label_dim, reuse = 1))

''' Decoder '''

# Decode_param = tf.global_variables()[24:30]

with tf.variable_scope('DZ'):
  ''' DiscrminatorZ '''
  DZreal1 = tf.nn.relu(nn_layer('DZ1', Z, dim1))
  DZreal2 = tf.nn.relu(nn_layer('DZ2', DZreal1, dim1))
  DZrealout = tf.nn.sigmoid(nn_layer('DZout', DZreal2, 1))
  DZ_param = tf.global_variables()[30:36]

  DZfake1 = tf.nn.relu(nn_layer('DZ1', z, dim1, reuse = 1))
  DZfake2 = tf.nn.relu(nn_layer('DZ2', DZfake1, dim1, reuse = 1))
  DZfakeout = tf.nn.sigmoid(nn_layer('DZout', DZfake2, 1, reuse = 1))
DZ_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DZ')

with tf.variable_scope('DY'):
  ''' DiscrminatorY '''
  DYreal1 = tf.nn.relu(nn_layer('DY1', Y, dim1))
  DYreal2 = tf.nn.relu(nn_layer('DY2', DYreal1, dim1))
  DYrealout = tf.nn.sigmoid(nn_layer('DYout', DYreal1, 1))
  DY_param = tf.global_variables()[36:42]

  DYfake1 = tf.nn.relu(nn_layer('DY1', y, dim1, reuse = 1))
  DYfake2 = tf.nn.relu(nn_layer('DY2', DYfake1, dim1, reuse = 1))
  DYfakeout = tf.nn.sigmoid(nn_layer('DYout', DYfake2, 1, reuse = 1))
DY_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DY')


# Print trainable variables
print("# Trainable variables")
params = tf.trainable_variables()
for param in params:
  print("  %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    ''' losses '''
    decoder_loss = 0.5*tf.reduce_mean(tf.pow(output-X,2))
   
    if square == False:
      DZ_loss = NLL * tf.reduce_mean(tf.log(DZrealout+eps) + tf.log(1. - DZfakeout + eps))
      GZ_loss = NLL * tf.reduce_mean(tf.log(DZfakeout + eps))
      DY_loss = NLL * tf.reduce_mean(tf.log(DYrealout+eps) + tf.log(1. - DYfakeout + eps))
      GY_loss = NLL * tf.reduce_mean(tf.log(DYfakeout + eps))
    else:
      DZ_loss = NLL * tf.reduce_mean(tf.pow(DZrealout-1, 2) + tf.pow(DZfakeout, 2))
      GZ_loss = NLL * tf.reduce_mean(tf.pow(DZfakeout-1, 2))
      DY_loss = NLL * tf.reduce_mean(tf.pow(DYrealout-1, 2) + tf.pow(DYfakeout, 2))
      GY_loss = NLL * tf.reduce_mean(tf.pow(DYfakeout-1, 2))
      
    tf.summary.scalar('AE_loss', decoder_loss)
    tf.summary.scalar('DZ_loss', DZ_loss)
    tf.summary.scalar('GZ_loss', GZ_loss)
    tf.summary.scalar('DY_loss', DY_loss)
    tf.summary.scalar('GY_loss', GY_loss)
    merged_summary = tf.summary.merge_all()
    G_loss = GZ_loss + GY_loss
    
    print(enc_params)
    print(gen_params)
    print(dec_params)
    print(DZ_params)
    print(DY_params)
    dec_solver_params = enc_params + gen_params + dec_params
    print('dec_solver_params = {}'.format(dec_solver_params))
    G_solver_params = enc_params + gen_params
    if use_adam == False:
      decode_solver = tf.train.MomentumOptimizer(lr1, 0.9).minimize(decoder_loss, var_list=dec_solver_params, global_step=global_step)
      DZ_solver = tf.train.MomentumOptimizer(lr2, 0.1).minimize(DZ_loss, var_list=DZ_params)
      DZ_solverLS = tf.train.MomentumOptimizer(lr2, 0.1).minimize(DZ_loss)
      DY_solver = tf.train.MomentumOptimizer(lr2, 0.1).minimize(DY_loss, var_list=DY_params)
      DY_solverLS = tf.train.MomentumOptimizer(lr2, 0.1).minimize(DY_loss)
      G_solver = tf.train.MomentumOptimizer(lr2, 0.1).minimize(G_loss, var_list=G_solver_params)
      G_solverLS = tf.train.MomentumOptimizer(lr2, 0.1).minimize(GZ_loss + GY_loss)
    else:
      decode_solver = tf.train.AdamOptimizer(0.001).minimize(decoder_loss, var_list=dec_solver_params)# + z_param + y_param)
      DZ_solver = tf.train.AdamOptimizer(0.001).minimize(DZ_loss, var_list=DZ_params)
      DY_solver = tf.train.AdamOptimizer(0.001).minimize(DY_loss, var_list=DY_params)
      G_solver = tf.train.AdamOptimizer(0.001).minimize(G_loss, var_list=G_solver_params)


saver = tf.train.Saver()


'''''''''''''''''''''''''''train'''''''''''''''''''''''''''''
p = 0

sess = tf.InteractiveSession()
if load == False:
  print('== new ==')
  sess.run(tf.global_variables_initializer())
else:
  print('== loaded == ')
  saver.restore(sess, os.path.join(out_dir, 'AuthorsUnsupervisedAAE.ckpt'))

mnist = input_data.read_data_sets(data_dir, one_hot=False)
X_test = mnist.train.images
Y_test = mnist.train.labels
n_data = len(mnist.train.images)

def get_batch():
  for i in range(0, n_data, mb_size):
    yield i, X_test[i:i+mb_size], Y_test[i:i+mb_size]

def run_eval():
  test_label = np.zeros([n_data, label_dim], dtype=np.float32)
  for i, x_batch, y_batch in get_batch():
    test_label[i:i+mb_size] = np.array(sess.run(y, feed_dict = {X: x_batch, keep_prob:1.0, phase:False}))
  return test_label

def calc_error_rate():
  test_label = run_eval()
  clusters = []
  for i in range(label_dim):
      clusters.append([])

  highest = np.zeros([label_dim, 2])

  for i in range(len(test_label)):
      label = np.argmax(test_label[i])
      if highest[label][1] < np.max(test_label[i]):
          highest[label][0] = i
          highest[label][1] = np.max(test_label[i])
      clusters[label].append(i)

  true_label = np.zeros(label_dim)
  for i in range(label_dim):
      true_label[i] = Y_test[int(highest[i][0])]
      
  error_count = 0
  for i in range(label_dim):
      for j in range(len(clusters[i])):
          if Y_test[clusters[i][j]] != true_label[i]:
              error_count += 1
              
  error_rate = error_count/len(X_test)
  print('error rate: {}'.format(error_rate))
  return error_rate

def get_global_step():
  return sess.run(global_step)

def add_summary(summary_writer, global_step, tag, value):
  """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., tag=BLEU.
  """
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
  summary_writer.add_summary(summary, global_step)

# train
summary_writer = tf.summary.FileWriter(summary_dir)

epoch_steps = math.ceil(n_data / mb_size)
print('n_data = {}'.format(n_data))
max_step = int(epoch_num*n_data/mb_size)
timer = Timer(max_step, period_second=15)
F_dec_loss, F_DZ_loss, F_DY_loss, F_GY_loss, F_GZ_loss = 9999, 9999, 7777, 7777, 7777
F_global_step = get_global_step()
for it in range(max_step):
    curr_epoch = (F_global_step+1) // epoch_steps
    learning_rate1 = 0.01
    learning_rate2 = 0.1 
    if curr_epoch  > change_LR_epoch:
      learning_rate1 = 0.001
      learning_rate2 = 0.01
        
        
    X_mb, _ = mnist.train.next_batch(mb_size)   # shuffled
    Y_mb = sample_Y(mb_size)
    Z_mb = sample_Z(mb_size)
    
    F_global_step, _, F_dec_loss = sess.run([global_step, decode_solver, decoder_loss], feed_dict={X:X_mb, lr1:learning_rate1, keep_prob:training_keep_prob, phase:turn_on_batch_norm})
    F_DZ_loss, _ = sess.run([DZ_loss, DZ_solver], feed_dict = {X:X_mb, Z:Z_mb, lr2:learning_rate2, keep_prob:1.0, phase:False})
    _, F_DY_loss, F_y = sess.run([DY_solver, DY_loss, y], feed_dict = {X:X_mb, Y:Y_mb, lr2:learning_rate2, keep_prob:1.0, phase:False})
    F_merged_summary, _, F_GY_loss, F_GZ_loss = sess.run([merged_summary, G_solver, GY_loss, GZ_loss], feed_dict = {X:X_mb, Y:Y_mb, Z:Z_mb, lr2:learning_rate2, keep_prob:training_keep_prob, phase:turn_on_batch_norm})

    summary_writer.add_summary(F_merged_summary, F_global_step)

    if F_global_step % 550 == 0: add_summary(summary_writer, F_global_step//550, 'epoch-error-rate', error_rate)

    
    if (F_global_step) % 100 == 0:
      timer.remaining(it)
      print('epoch {}   step = {} / {}'.format(curr_epoch, F_global_step, max_step))
      print('LR1 = {}   dec_loss = {:.5f}  DZ_loss = {:.5f}   DY_loss = {:.5f}    GY_loss = {:.5f}    GZ_loss = {:.5f}'.format(learning_rate1, F_dec_loss, F_DZ_loss, F_DY_loss, F_GY_loss, F_GZ_loss))
      error_rate = calc_error_rate()
      add_summary(summary_writer, F_global_step, 'step-error-rate', error_rate)

    
    '''******** Generate Sample Images ********'''
    if it % 1000 == 0:
        error_rate = calc_error_rate()
        print('Iter: {}'.format(F_global_step))
        n_sample = 16
        
        fig = plot(X_mb[:n_sample])
        plt.savefig(os.path.join(out_dir, '{}in.png'.format(str(p).zfill(3))), bbox_inches='tight')
        
        plt.close(fig)
        
        
        samples = sess.run(output[:n_sample], feed_dict={X:X_mb, keep_prob:1.0, phase:False})
        
        fig = plot(samples)
        plt.savefig(os.path.join(out_dir, '{}out.png'.format(str(p).zfill(3))), bbox_inches='tight')
        
        plt.close(fig)
        p += 1
        saver.save(sess, os.path.join(out_dir, 'AuthorsUnsupervisedAAE.ckpt'))
        print('saved')


''''''''''''''''''''''''''''test'''''''''''''''''''''''''''
for k in range(label_dim):
    if not os.path.exists(os.path.join(out_dir,  '{}AAEout'.format(str(k+1).zfill(2)))):
        os.makedirs(os.path.join(out_dir, '{}AAEout'.format(str(k+1).zfill(2))))
  

calc_error_rate()


print((X_test).shape)
print('==== make images of clustering results ===')
test_label = run_eval()
for i in range(len(X_test[:1000])):
    target = np.argmax(test_label[i]) + 1
    if fast_img_save == False:
      fig = plot2(X_test[i])
      plt.savefig(os.path.join(out_dir, '{}AAEout'.format(str(target).zfill(2)), '{}.png'.format(str(i).zfill(5), bbox_inches='tight')))
      plt.close(fig)
    else:
       matplotlib.image.imsave(os.path.join(out_dir, '{}AAEout'.format(str(target).zfill(2)), '{}.png'.format(str(i).zfill(5), bbox_inches='tight')), X_test[i].reshape(28, 28))


