__author__ = "Shreyas Seshadri, shreyas.seshadri@aalto.fi"


import numpy as np
import tensorflow as tf


tf.reset_default_graph() # debugging, clear all tf variables
#tf.enable_eager_execution() # placeholders are not compatible

import model_convNetIN 
import scipy.io
import math
import time

_FLOATX = tf.float32 



## LOAD DATA

loadFile = './pyDatGit.mat'
loaddata = scipy.io.loadmat(loadFile)
X = loaddata['X']
Y = loaddata['Y']


## PARAMETERS
z_dim = 2
residual_channels = 256
filter_width = 11
dilations = [1]*6
input_channels = X.shape[2]
output_channels = X.shape[2]
cond_dim = z_dim
postnet_channels= 256
do_postproc = False
do_glu = True

#z_dim = 4
#residual_channels = 2
#filter_width = 3
#dilations = [1]
#input_channels = X[0][0].shape[2]
#output_channels = X[0][0].shape[2]
#cond_dim = z_dim
#postnet_channels= 5
#do_postproc = False
#do_glu = True
is_train = tf.placeholder( dtype=tf.bool)
G_xy = model_convNetIN.CNET(name='G_xy', 
                       input_channels=input_channels,
                       output_channels= output_channels,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       cond_channels = cond_dim,
                       postnet_channels=postnet_channels,
                       cond_dim=cond_dim,
                       do_postproc=do_postproc,
                       do_GLU=do_glu,
                       isTrain=is_train)

G_yx = model_convNetIN.CNET(name='G_yx', 
                       input_channels=input_channels,
                       output_channels= output_channels,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=None,
                       do_postproc=do_postproc,
                       do_GLU=do_glu,
                       isTrain=is_train)

E_zy = model_convNetIN.CNET(name='E_zy', 
                       input_channels=input_channels+input_channels,
                       output_channels= z_dim,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=None,
                       do_postproc=do_postproc,
                       do_GLU=do_glu,
                       isTrain=is_train)

D_x = model_convNetIN.CNET(name='D_x', 
                       input_channels=output_channels,
                       output_channels= 1,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=None,
                       do_postproc=do_postproc,
                       do_GLU=do_glu,
                       isTrain=is_train)

D_y = model_convNetIN.CNET(name='D_y', 
                       input_channels=output_channels,
                       output_channels= 1,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=None,
                       do_postproc=do_postproc,
                       do_GLU=do_glu,
                       isTrain=is_train)

D_zy = model_convNetIN.CNET(name='D_zy', 
                       input_channels=z_dim,
                       output_channels= 1,
                       residual_channels=8,
                       filter_width=1,
                       dilations=[1],
                       postnet_channels=8,
                       cond_dim=None,
                       do_postproc=do_postproc,
                       do_GLU=do_glu,
                       isTrain=is_train)

# optimizer parameters
adam_lr = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999

num_iters = 5000#3#200

# data placeholders of shape (batch_size, timesteps, feature_dim)
x_real = tf.placeholder(shape=(None, None, input_channels), dtype=_FLOATX)
y_real = tf.placeholder(shape=(None, None, input_channels), dtype=_FLOATX)
z_y = tf.placeholder(shape=(None, None, cond_dim), dtype=_FLOATX)
z_yid = tf.placeholder(shape=(None, None, cond_dim), dtype=_FLOATX)
id_scale = tf.placeholder(shape=(None), dtype=_FLOATX)


# X -> Y_hat -> X_hat_hat loop
y_hat = G_xy.forward_pass(x_real, cond_input=z_y)
x_hat_hat = G_yx.forward_pass(y_hat)
z_y_hat_hat = E_zy.forward_pass(tf.concat([x_real,y_hat],axis=2))
x_id = G_yx.forward_pass(x_real)

D_out_y_real = D_y.forward_pass(y_real)
D_out_y_fake = D_y.forward_pass(y_hat)

# Y -> X_hat -> Y_hat_hat loop
x_hat = G_yx.forward_pass(y_real)
z_y_hat = E_zy.forward_pass(tf.concat([x_hat,y_real],axis=2))
y_hat_hat = G_xy.forward_pass(x_hat, cond_input=z_y_hat)
y_id = G_xy.forward_pass(y_real, cond_input=z_yid)

D_out_x_real = D_x.forward_pass(x_real)
D_out_x_fake = D_x.forward_pass(x_hat)

D_out_z_fake = D_zy.forward_pass(z_y_hat)
D_out_z_real = D_zy.forward_pass(z_y)


# GAN loss
D_y_loss_gan = -tf.reduce_mean(D_out_y_real) + tf.reduce_mean(D_out_y_fake) 
D_x_loss_gan = -tf.reduce_mean(D_out_x_real) + tf.reduce_mean(D_out_x_fake) 
D_zy_loss_gan = -tf.reduce_mean(D_out_z_real) + tf.reduce_mean(D_out_z_fake) 
 
G_xy_loss_gan = -tf.reduce_mean(D_out_y_fake)
G_yx_loss_gan = -tf.reduce_mean(D_out_x_fake)
E_zy_loss_gan = -tf.reduce_mean(D_out_z_fake)

#recon loss
recon_loss_x = 10*tf.reduce_mean(tf.abs(x_real - x_hat_hat)) 
recon_loss_y = 10*tf.reduce_mean(tf.abs(y_real - y_hat_hat))
recon_loss_z = 10*tf.reduce_mean(tf.abs(z_y - z_y_hat_hat))


# identity loss
id_loss_x = id_scale*tf.reduce_mean(tf.abs(x_real-x_id))
id_loss_y = id_scale*tf.reduce_mean(tf.abs(y_real-y_id))

# gradient penalty
epsilon_shape = tf.stack([tf.shape(x_real)[0],tf.shape(x_real)[1],1]) # treat timestep similar to batch (TODO: or not?)
epsilon = tf.random_uniform(epsilon_shape, 0.001, 0.999)
y_grad = epsilon*y_real + (1.0-epsilon)*y_hat
d_hat_y = D_y.forward_pass(y_grad)
gradients_y = tf.gradients(d_hat_y, y_grad)[0]
gradnorm_y = tf.sqrt(tf.reduce_sum(tf.square(gradients_y), axis=[2])+1.0e-19)
gradient_penalty_y = 10*tf.reduce_mean(tf.square(gradnorm_y-1.0)) # magic weight factor 10 from the 'improved wgan' paper
D_loss_gradpen_y = gradient_penalty_y

epsilon_shape = tf.stack([tf.shape(y_real)[0],tf.shape(x_real)[1],1]) # treat timestep similar to batch (TODO: or not?)
epsilon = tf.random_uniform(epsilon_shape, 0.001, 0.999)
x_grad = epsilon*x_real + (1.0-epsilon)*x_hat
d_hat_x = D_x.forward_pass(x_grad)
gradients_x = tf.gradients(d_hat_x, x_grad)[0]
gradnorm_x = tf.sqrt(tf.reduce_sum(tf.square(gradients_x), axis=[2])+1.0e-19)
gradient_penalty_x = 10*tf.reduce_mean(tf.square(gradnorm_x-1.0)) # magic weight factor 10 from the 'improved wgan' paper
D_loss_gradpen_x = gradient_penalty_x

epsilon_shape = tf.stack([tf.shape(z_y)[0],tf.shape(z_y)[1],1]) # treat timestep similar to batch (TODO: or not?)
epsilon = tf.random_uniform(epsilon_shape, 0.001, 0.999)
z_y_grad = epsilon*z_y + (1.0-epsilon)*z_y_hat
d_hat_zy = D_zy.forward_pass(z_y_grad)
gradients_zy = tf.gradients(d_hat_zy, z_y_grad)[0]
gradnorm_zy = tf.sqrt(tf.reduce_sum(tf.square(gradients_zy), axis=[2])+1.0e-19)
gradient_penalty_zy = 10*tf.reduce_mean(tf.square(gradnorm_zy-1.0)) # magic weight factor 10 from the 'improved wgan' paper
D_loss_gradpen_zy = gradient_penalty_zy


# additional penalty term to keep the scores from drifting too far from zero 
D_loss_zeropen_x = 1e-2 * tf.reduce_sum(tf.square(D_out_x_real))
D_loss_zeropen_y = 1e-2 * tf.reduce_sum(tf.square(D_out_y_real))
D_loss_zeropen_zy = 1e-2 * tf.reduce_sum(tf.square(D_out_z_real))

D_loss = D_y_loss_gan + D_x_loss_gan + D_zy_loss_gan + D_loss_gradpen_y + D_loss_gradpen_x + D_loss_gradpen_zy + D_loss_zeropen_x + D_loss_zeropen_y + D_loss_zeropen_zy
Gen_loss = G_xy_loss_gan + G_yx_loss_gan + E_zy_loss_gan + recon_loss_x + recon_loss_y + recon_loss_z + id_loss_x + id_loss_y
Gen_loss2 = G_xy_loss_gan + G_yx_loss_gan + E_zy_loss_gan + recon_loss_x + recon_loss_y + recon_loss_z

theta_G_xy = G_xy.get_variable_list()
theta_G_yx = G_yx.get_variable_list()
theta_E_zy = E_zy.get_variable_list()
theta_Dx = D_x.get_variable_list()
theta_Dy = D_y.get_variable_list()
theta_D_zy = D_zy.get_variable_list()

Gen_solver = tf.train.AdamOptimizer(learning_rate=adam_lr,beta1=adam_beta1,beta2=adam_beta2).minimize(Gen_loss, var_list=[theta_G_xy,theta_G_yx,theta_E_zy])

D_solver = tf.train.AdamOptimizer(learning_rate=adam_lr,beta1=adam_beta1,beta2=adam_beta2).minimize(D_loss, var_list=[theta_Dx,theta_Dy,theta_D_zy])


n_critic = 5

lossD_all = np.zeros((num_iters,10),dtype=float)
lossG_all = np.zeros((num_iters,9),dtype=float)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
tfconfig = tf.ConfigProto(gpu_options=gpu_options)

saveFile1 = './pred_res.mat'
saveFile2 = './errors.mat'

model_path = './model.ckpt'
saver = tf.train.Saver()

n_frames = X.shape[1]
batch_size = 32
const_id_sc = 2500
id_sc = np.concatenate((np.ones(const_id_sc)*5,np.linspace(5.0,0.001,num_iters-const_id_sc)),axis=0)

x_size = X[0].shape[0]

cont = 0;
with tf.Session(config=tfconfig) as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op) 

    saver = tf.train.Saver(max_to_keep=0)

    for it in range(num_iters):
        t = time.time()
        # Train discriminator
        
        X_tmp = X[np.random.randint(X.shape[0],size=batch_size)]
        Y_tmp = Y[np.random.randint(Y.shape[0],size=batch_size)]
        
        
        for critic_i in range(n_critic):

            # Train Discriminator 
            
            rand_val_1 = np.random.randn(batch_size, z_dim)
            rand_val_1 = np.tile(np.reshape(rand_val_1,[rand_val_1.shape[0],1,rand_val_1.shape[1]]), [1, n_frames, 1])
            rand_val_2 = np.random.randn(batch_size, z_dim)
            rand_val_2 = np.tile(np.reshape(rand_val_2,[rand_val_1.shape[0],1,rand_val_2.shape[1]]), [1, n_frames, 1])
            _, lossD, lossD_gan_x, lossD_gan_y, lossD_gan_zy, lossD_grad_y, lossD_grad_x, lossD_grad_zy, lossD_zero_x, lossD_zero_y, lossD_zero_zy = sess.run([D_solver, D_loss, D_x_loss_gan, D_y_loss_gan,D_zy_loss_gan,D_loss_gradpen_y, D_loss_gradpen_x, D_loss_gradpen_zy, D_loss_zeropen_x, D_loss_zeropen_y,D_loss_zeropen_zy], feed_dict={x_real: X_tmp,y_real: Y_tmp, z_y:rand_val_1, z_yid:rand_val_2,id_scale:id_sc[it],is_train:True})
        lossD_all[cont][0] = lossD
        lossD_all[cont][1] = lossD_gan_x
        lossD_all[cont][2] = lossD_gan_y
        lossD_all[cont][3] = lossD_gan_zy
        lossD_all[cont][4] = lossD_grad_y
        lossD_all[cont][5] = lossD_grad_x
        lossD_all[cont][6] = lossD_grad_zy
        lossD_all[cont][7] = lossD_zero_x
        lossD_all[cont][8] = lossD_zero_y
        lossD_all[cont][9] = lossD_zero_zy
        
        # Train generator
        rand_val_1 = np.random.randn(batch_size, z_dim)
        rand_val_1 = np.tile(np.reshape(rand_val_1,[rand_val_1.shape[0],1,rand_val_1.shape[1]]), [1, n_frames, 1])
        rand_val_2 = np.random.randn(batch_size, z_dim)
        rand_val_2 = np.tile(np.reshape(rand_val_2,[rand_val_1.shape[0],1,rand_val_2.shape[1]]), [1, n_frames, 1])
        _, lossGen, lossG_xy, lossG_yx, lossE_zy, loss_reconX, loss_reconY, loss_reconZY, loss_idX, loss_idY = sess.run([Gen_solver, Gen_loss, G_xy_loss_gan, G_yx_loss_gan, E_zy_loss_gan, recon_loss_x, recon_loss_y, recon_loss_z, id_loss_x, id_loss_y], feed_dict={x_real: X_tmp,y_real: Y_tmp, z_y:rand_val_1, z_yid:rand_val_2,id_scale:id_sc[it],is_train:True})
        
        lossG_all[cont][0] = lossGen
        lossG_all[cont][1] = lossG_xy
        lossG_all[cont][2] = lossG_yx
        lossG_all[cont][3] = lossE_zy
        lossG_all[cont][4] = loss_reconX
        lossG_all[cont][5] = loss_reconY
        lossG_all[cont][6] = loss_reconZY
        lossG_all[cont][7] = loss_idX
        lossG_all[cont][8] = loss_idY
        cont = cont+1
                
            #print("Errors for epoch %d and minibatch %d : Gen loss is %f, D loss is %f " % (epoch, batch_i, lossGen, lossD))
        elapsed = time.time() - t    
        print("Errors for minibatch %d : Gen loss is %f, D loss is %f  and took time %f" % (it, lossGen, lossD,elapsed))
        scipy.io.savemat(saveFile2,{"lossG_all":lossG_all,"lossD_all":lossD_all})
        
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)    
    

    
