"""
********************************************************************************
deep neural network
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
        self.f_in   = f_in     # input features
        self.f_out  = f_out    # output features
        self.f_hid  = f_hid    # hidden layer dim
        self.depth  = depth    # depth of dnn
        self.w_init = w_init   # weight initializer
        self.b_init = b_init   # bias initializer
        self.act    = act      # activation
        self.lr     = lr       # learning rate
        self.opt    = opt      # optimizer ("SGD" / "RMSprop" / "Adam")
        self.f_scl  = f_scl    # feature scaling ("linear" / "minmax" / "mean")
        self.d_type = d_type   # data type
        self.r_seed = r_seed   # random seed

        # set data type and random seed
        self.type_seed(self.d_type, self.r_seed)

        # input - output pair
        self.x = x
        self.y = y

        XY = tf.concat([x, y], 1)
        self.lower = tf.cast(tf.reduce_min(XY, axis = 0), dtype=self.d_type)
        self.upper = tf.cast(tf.reduce_max(XY, axis = 0), dtype=self.d_type)
        self.mean = tf.cast(tf.reduce_mean(XY, axis = 0), dtype=self.d_type)

        # build a deep neural network
        self.w_init = self.weight_init(self.w_init, self.r_seed)
        self.b_init = self.bias_init(self.b_init)
        self.act    = self.act_func(self.act)
        self.dnn = self.dnn_init(
            self.f_in, self.f_out, self.f_hid, self.depth, 
            w_init, b_init, act
        )
        self.optimizer = self.opt_alg(self.lr, self.opt)
        self.loss_log = []

        # print some key settings
        print("\n************************************************************")
        print("****************     MAIN PROGRAM START     ****************")
        print("************************************************************")

    def type_seed(
        self, d_type, r_seed
    ):
        print("\n>>>>> type_seed")
        print("         data type  :", d_type)
        print("         random seed:", r_seed)
        os.environ["PYTHONHASHSEED"] = str(r_seed)
        np.random.seed(r_seed)
        tf.random.set_seed(r_seed)
        tf.keras.backend.set_floatx(d_type)

    def weight_init(
        self, init, seed
    ):
        print("\n>>>>> weight_init")
        print("         initializer:", init)
        if init == "Glorot":
            weight = tf.keras.initializers.GlorotNormal(seed = seed)
        elif init == "He":
            weight = tf.keras.initializers.HeNormal(seed = seed)
        elif init == "LeCun":
            weight = tf.keras.initializers.LecunNormal(seed = seed)
        else:
            raise NotImplementedError(">>>>> weight_init")
        return weight

    def bias_init(
        self, init
    ):
        print("\n>>>>> bias_init")
        print("         initializer:", init)
        if init == "zeros":
            bias = tf.keras.initializers.Zeros()
        elif init == "ones":
            bias = tf.keras.initializers.Ones()
        else:
            raise NotImplementedError(">>>>> bias_init")
        return bias

    def act_func(
        self, act
    ):
        print("\n>>>>> act_func")
        print("         activation:", act)
        if act == "relu":
            activation = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0.0)
        return activation

    def dnn_init(
        self, 
        f_in, f_out, f_hid, depth,
        w_init, b_init, act
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
            dnn.add(tf.keras.layers.Lambda(
                lambda x: 2. * (x - self.lower) / (self.upper - self.lower) - 1.
            ))
        elif self.f_scl == "mean":
            dnn.add(tf.kears.layers.Lambda(
                lambda x: (x - self.mean) / (self.upper - self.lower)
            ))
        else:
            raise NotImplementedError(">>>>> dnn_init")
        for l in range(depth - 1):
            dnn.add(
                tf.keras.layers.Dense(
                    f_hid, activation = act, use_bias = True, 
                    kernel_initializer = w_init, bias_init = b_init, 
                    kernel_regularizer = None, bias_regularizer = None, 
                    activity_regularizer = None, kernel_constraint = None, bias_constraint = None
                )
            )
        dnn.add(tf.keras.layers.Dense(f_out, activation = "linear"))
        return dnn

    def opt_alg(
        self, lr, opt
    ):
        print("\n>>>>> opt_alg")
        print("         learning rate:", lr)
        print("         optimizer    :", opt)
        if opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.0, nesterov = False)
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
        self, n_epoch, n_batch, c_tlrnc
    ):
        print("\n>>>>> train")
        print("         n_epoch:", n_epoch)
        print("         n_batch:", n_batch)
        print("         c_tlrnc:", c_tlrnc)

        if n_batch == -1:
            print("\n>>>>> executing full-batch training")
            for epc in range(n_epoch):

        else:
            print("\n>>>>> executing mini-batch training")
            for epc in range(n_epoch):
                for idx in range(n_bs):



    def infer(
        self, x
    ):
        print("\n>>>>> infer")

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


