from models.a2c import *
#import tensorflow as tf
class ppo(a2c):
    def __init__(self):
        #a2c.__init__(self) #we dont want this graph
        ## let's change the graph according to ppo
        epsil=0.3
        tf.reset_default_graph()
        self.observation=tf.placeholder(tf.float32, shape=input_shape)
        self.R= tf.placeholder(tf.float32,shape=[None,1]) #not immediate but n step discounted
        self.a_t=tf.placeholder(tf.float32,shape=[None,no_of_actions]) #which action was taken 
        self.p_logit,self.pi_params= self.actor(self.observation,'pi',trainable=True)#probabilities of action predicted
        self.p=tf.nn.softmax(self.p_logit,name='new_prob')
        self.oldp_logit,self.oldpi_params=self.actor(self.observation,'pi_old',trainable=False)
        self.oldp=tf.nn.softmax(self.oldp_logit,name='old_prob')
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op=[oldp.assign(p) for p,oldp in zip(self.pi_params, self.oldpi_params)]
        self.V= self.critic(self.observation) #value predicted

        #saver
        self.saver = tf.train.Saver()

        self.advantage= self.R - self.V
        #surrogate, clipping method and losses
        with tf.variable_scope('loss'):
            self.exec_prob=self.p*self.a_t
            self.old_prob=self.oldp*self.a_t
            self.ratio=tf.reduce_sum(self.exec_prob, axis=1) /tf.reduce_sum(self.old_prob)
            self.surr=self.ratio*self.advantage
            self.loss_policy = - tf.reduce_sum(tf.minimum(self.surr,tf.clip_by_value(self.ratio,1.0-epsil,1.0+epsil)*self.advantage))
            self.loss_value  = LOSS_V * tf.nn.l2_loss(self.advantage)               # minimize value error
        #will try entropy loss afterwards
            self.entropy =  tf.reduce_sum(self.p * tf.log(self.p + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        #self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value + self.entropy)
        with tf.variable_scope('optimizers'):
            self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(self.loss_policy+self.entropy)
            self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(self.loss_value)
        
        #session and initialization
        self.sess=tf.Session()
        #writers
        self.writer = tf.summary.FileWriter(tf_logdir, self.sess.graph)

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init_op)
        self.default_graph=tf.get_default_graph()
        self.default_graph.finalize()
    ## override action method here so that it should return params for updating pi_old with pi
    def actor(self,inputs,name,trainable):
        actions=net(inputs,10,no_of_actions,name,trainable)
        params=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return actions,params
