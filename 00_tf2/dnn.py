"""
********************************************************************************
deep neural dnn
********************************************************************************
"""

import os
import time
import numpy as np
import tensorflow as tf

class dnn_1D(tf.keras.Model):
    def __init__(
        self, 
        x, y, 
        f_in, f_out, f_hid, depth, 
        w_init, b_init, act, 
        lr = 5e-4, opt = "Adam", f_scl = "minmax", 
        d_type = "float32", r_seed = 1234
    ):
        # initialization
        self.f_in   = f_in
        self.f_out  = f_out
        self.f_hid  = f_hid
        self.depth  = depth
        self.w_init = w_init
        self.b_init = b_init
        self.act    = act
        self.lr     = lr
        self.opt    = opt
        self.f_scl  = f_scl
        self.d_type = d_type
        self.r_seed = r_seed

        # set data type and random seed
        self.setup(self.d_type, self.r_seed)

        # input - output pair
        self.x = x
        self.y = y

        # build a deep neural network
        self.dnn = self.dnn_init(self.f_in, self.f_out, self.f_hid, self.depth)
        self.optimizer = self.opt_alg(self.lr, self.opt)
        self.loss_log = []

        # print some key settings
        print("\n************************************************************")
        
        print("****************     MAIN PROGRAM START     ****************")

        print("   ")
        print("************************************************************")


        print("************************************************************")

    def setup(
        self, d_type, r_seed
    ):
        print("\n>>>>> setup")
        print("         data type  :", d_type)
        print("         random seed:", r_seed)
        os.environ["PYTHONHASHSEED"] = str(r_seed)
        np.random.seed(r_seed)
        tf.random.set_seed(r_seed)
        tf.keras.backend.set_floatx(d_type)

    def dnn_init(
        self, f_in, f_out, f_hid, depth
    ):
        print("\n>>>>> dnn_init")
        print("         f_in :", f_in)
        print("         f_out:", f_out)
        print("         f_hid:", f_hid)
        print("         depth:", depth)
        dnn = tf.keras.Sequential()
        dnn.add(tf.keras.layers.InputLayer(f_in))
        if self.f_scl == "linear":
            dnn.add(tf.keras.layers.Lambda(lambda x: x))
        elif self.f_scl == "minmax":
            dnn.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))
        elif self.f_scl == "mean":
            dnn.add(tf.keras.layers.Lambda(lambda x: (x - self.mn) / (self.ub - self.lb)))
        else:
            raise NotImplementedError(">>>>> dnn_init")
        for l in range(depth - 1):
            dnn.add(
                tf.keras.layers.Dense(
                    f_hid, activation = self.act, use_bias = True, 
                    kernel_initializer = self.w_init, bias_initializer = self.b_init, 
                    kernel_regularizer = None, bias_regularizer = None, 
                    activity_regularizer = None, kernel_constraint = None, bias_constraint = None
                )
            )
        dnn.add(tf.keras.layers.Dense(f_out, activation = "linear"))
        return dnn

    def act_func(
        self, act
    ):
        print("\n>>>>> act_func")
        print("         activation:", act)
        if act == "relu":
            activation = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
        return activation

    def opt_alg(
        self, lr, opt
    ):
        print("\n>>>>> opt_alg")
        print("         learning rate:", lr)
        print("         optimizer    :", opt)
        if opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.0, nesterov = False)
        elif opt == "Momentum":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.9, nesterov = False)
        elif opt == "NAG":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.9, nesterov = True)
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False)
        elif opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
        elif opt == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        elif opt == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        else:
            raise NotImplementedError(">>>>> opt_alg")
        return optimizer

    def train(
        self, n_epoch, n_batch, c_tol
    ):
        print("\n>>>>> train")
        print("         n_epoch:", n_epoch)
        print("         n_batch:", n_batch)
        print("         c_tol  :", c_tol)

        # full-batch training
        print("\n>>>>> executing full-batch training")
        for 
        # mini-batch training
        print("\n>>>>> executing mini-batch training")
        for 

# class dnn_2D(tf.keras.Model):
#     def __init__(
#         self, 
#         x, y, z, 
#         f_in, f_out, f_hid, depth, 
#         w_init, b_init, act, 
#         lr = 5e-4, opt = "Adam", f_scl = "minmax", 
#         d_type = "float32", r_seed = 1234
#     ):
#         # initialization


