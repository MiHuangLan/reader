# -*- coding: utf-8 -*-


import os
import time
import logging
import json
import numpy as np

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, position_embedding
from optimizer import AdamWOptimizer
from tensorflow.python.ops import array_ops

from utils.dureader_eval import compute_bleu_rouge
from utils.dureader_eval import normalize
from utils.basic_rnn import rnn
from utils.match_layer import AttentionFlowMatchLayer
from utils.pointer_net import PointerNetDecoder

class Model(object):
    def __init__(self, vocab, args, demo=False):

        # logging
        self.logger = logging.getLogger("QANet")
        # self.config = config
        self.demo = demo

        # basic config
        self.algo_match = args.algo_match
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout < 1
        self.batch_size=args.batch_size
        self.fix_pretrained_vector=args.fix_pretrained_vector
        self.use_position_attn=args.use_position_attn
        self.clip_weight=args.clip_weight
        self.max_norm_grad=args.max_norm_grad
        self.char_embed_size=args.char_embed_size
        self.head_size=args.head_size
        self.algo=args.algo
        self.loss_type=args.loss_type
        self.decay=args.decay
        # length limit
        if not self.demo:
            self.max_p_num = args.max_p_num
            self.logger.info("numbers of passages %s" % self.max_p_num)
        else:
            self.max_p_num = 1

        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len
        self.max_ch_len=args.max_ch_len
        self.hidden_size=args.hidden_size
        # the vocab
        self.l2_norm=args.l2_norm
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=sess_config)
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        #self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._self_attention()
        self._mul_attention()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = total_params(tf.trainable_variables())
        self.logger.info('There are {} parameters in the model'.format(param_num))

    """
    :description: Placeholders
    """
    def _setup_placeholders(self):
        
        if self.demo:
            self.c = tf.placeholder(tf.int32, [None, self.max_p_len],"context")
            self.q = tf.placeholder(tf.int32, [None, self.max_q_len],"question")
            self.ch = tf.placeholder(tf.int32, [None, self.max_p_len, self.max_ch_len], "context_char")
            self.qh = tf.placeholder(tf.int32, [None, self.max_q_len, self.max_ch_len], "question_char")
            self.start_label = tf.placeholder(tf.int32, [None],"answer_label1")
            self.end_label = tf.placeholder(tf.int32, [None],"answer_label2")
        else:
            self.c = tf.placeholder(tf.int32, [self.batch_size*self.max_p_num, self.max_p_len], "context")
            self.q = tf.placeholder(tf.int32, [self.batch_size*self.max_p_num, self.max_q_len], "question")
            self.ch = tf.placeholder(tf.int32, [self.batch_size*self.max_p_num, self.max_p_len, self.max_ch_len], "context_char")
            self.qh = tf.placeholder(tf.int32, [self.batch_size*self.max_p_num, self.max_q_len, self.max_ch_len], "question_char")
            self.start_label = tf.placeholder(tf.int32, [self.batch_size],"answer_label1")
            self.end_label = tf.placeholder(tf.int32, [self.batch_size],"answer_label2")

        self.position_emb = position_embedding(self.c, 2*self.hidden_size)
        self.c_mask = tf.cast(self.c, tf.bool) # index 0 is padding symbol  N x self.max_p_num, max_p_len
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        self.dropout = tf.placeholder(tf.float32, name="dropout")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

    """
    :descrition: The embedding layer, question and passage share embeddings
    """

    def _embed(self):
        with tf.variable_scope('word_char_embedding'):
            if self.fix_pretrained_vector:
                self.pretrained_word_mat = tf.get_variable("word_emb_mat", 
                              	[self.vocab.word_size() - 2, self.vocab.word_embed_dim], 
	                            dtype=tf.float32,
	                            initializer=tf.constant_initializer(self.vocab.word_embeddings[2:],
	                                                                dtype=tf.float32),
	                            trainable=False)
                self.word_pad_unk_mat = tf.get_variable("word_unk_pad", 
	                            [2, self.pretrained_word_mat.get_shape()[1]],
	                            dtype=tf.float32,
	                            initializer=tf.constant_initializer(self.vocab.word_embeddings[:2], 
	                                                    dtype=tf.float32),
	                            trainable=True)

                self.word_mat = tf.concat([self.word_pad_unk_mat, self.pretrained_word_mat], axis=0)

                self.pretrained_char_mat = tf.get_variable("char_emb_mat", 
                              	[self.vocab.char_size() - 2, self.vocab.char_embed_dim], 
                              	dtype=tf.float32,
                              	initializer=tf.constant_initializer(self.vocab.char_embeddings[2:], 
                                                                dtype=tf.float32),
                              	trainable=False)
                self.char_pad_unk_mat = tf.get_variable("char_unk_pad", 
                                [2, self.pretrained_char_mat.get_shape()[1]],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(self.vocab.char_embeddings[:2], 
                                                        dtype=tf.float32),
                                trainable=True)

                self.char_mat = tf.concat([self.char_pad_unk_mat, self.pretrained_char_mat], axis=0)

            else:
                self.word_mat = tf.get_variable(
                                'word_embeddings',
                                shape=[self.vocab.word_size(), self.vocab.word_embed_dim],
                                initializer=tf.constant_initializer(self.vocab.word_embeddings),
                                trainable=True
                )

                self.char_mat = tf.get_variable(
                                'char_embeddings',
                                shape=[self.vocab.char_size(), self.vocab.char_embed_dim],
                                initializer=tf.constant_initializer(self.vocab.char_embeddings),
                                trainable=True
                )

            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        N, PL, QL, CL, d, dc, nh = self._params()

        if self.fix_pretrained_vector:
            dc = self.char_mat.get_shape()[-1]
        with tf.variable_scope("Input_Embedding_Layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.ch), [self.batch_size*self.max_p_len*self.max_p_num, self.max_ch_len, self.char_embed_size])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.qh), [self.batch_size*self.max_q_len*self.max_p_num, self.max_ch_len, self.char_embed_size])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            ch_emb = conv(ch_emb, d,
                bias = True, activation = tf.nn.tanh, kernel_size = 5, name = "char_conv", reuse = None)
            qh_emb = conv(qh_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = True)

            ch_emb = tf.reduce_max(ch_emb, axis = 1)
            qh_emb = tf.reduce_max(qh_emb, axis = 1)

            ch_emb = tf.reshape(ch_emb, [N* self.max_p_num, PL, -1])
            qh_emb = tf.reshape(qh_emb, [N* self.max_p_num, QL, -1])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            self.c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)
            self.q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)

    def _encode(self):
        N, PL, QL, CL, d, dc, nh = self._params()
        if self.fix_pretrained_vector:
            dc = self.char_mat.get_shape()[-1]
        with tf.variable_scope('passage_encoding'):
            self.sep_c_encodes, _ = rnn('bi-gru', self.c_emb, self.c_len, self.hidden_size)
        self.sc=self.sep_c_encodes
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-gru', self.q_emb, self.q_len, self.hidden_size)
        # if self.use_dropout:
        #     self.sep_c_encodes = tf.nn.dropout(self.sep_c_encodes, self.dropout)
        #     self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        N, PL, QL, CL, d, dc, nh = self._params()
        if self.fix_pretrained_vector:
            dc = self.char_mat.get_shape()[-1]
        if self.algo_match == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo_match == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_c_encodes, self.sep_q_encodes,
                                                    self.c_len, self.q_len)
        self.mp=self.match_p_encodes
        # if self.use_dropout:
        #     self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        N, PL, QL, CL, d, dc, nh = self._params()
        if self.fix_pretrained_vector:
            dc = self.char_mat.get_shape()[-1]

        with tf.variable_scope('fusion'):
            # print('match_p###############')
            # print(self.match_p_encodes.get_shape())
            self.fuse_p_encodes, _ = rnn('bi-gru', self.match_p_encodes,self.c_len,
                                         self.hidden_size, layer_num=1)
            self.f=self.fuse_p_encodes
            # if self.use_dropout:
            #     self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout)
            # # print('!!!!!!!!!!!!!######')
            # print(self.fuse_p_encodes.get_shape())

            #self.fuse_p_encodes = tf.reshape(self.fuse_p_encodes, [-1, 2 * self.hidden_size])

    def _self_attention(self):
        with tf.variable_scope('self_attion'):

            W = tf.get_variable(name="attn_W",
                                shape=[self.batch_size*self.max_p_num, 2*self.hidden_size, 2*self.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            V = tf.get_variable(name="attn_V", shape=[self.batch_size*self.max_p_num,2* self.hidden_size, 1],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            U = tf.get_variable(name="attn_U",
                                shape=[self.batch_size*self.max_p_num,2*self.hidden_size, 2*self.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            shape = tf.shape(self.fuse_p_encodes)
            atten_hidden = tf.tanh(
                tf.add(
                    tf.matmul(self.fuse_p_encodes, W),
                    tf.matmul(self.fuse_p_encodes, U)))
            # print('atten_h###############')
            # print(atten_hidden.get_shape())
            max_c = tf.reduce_max(tf.reshape(tf.matmul(atten_hidden, V), [-1, shape[1], 1]))
            alpha = tf.nn.softmax(
                #tf.reshape(tf.matmul(atten_hidden, V), [-1, shape[1], 1])-max_c, axis=1)
                tf.reshape(tf.matmul(atten_hidden, V), [-1, shape[1], 1])-max_c)
            # print('softmax###############')
            # print(alpha.get_shape())
            output = tf.reshape(self.fuse_p_encodes, [-1, shape[1], 2*self.hidden_size])
            C = tf.multiply(alpha, output)
            # print('2###############')
            # print(C.get_shape())
            output_C=tf.concat([output, C], axis=-1)
            # print('output_c###############')
            # print(output_C.get_shape())

            self.anttion_p, _ = rnn('bi-gru', output_C, self.c_len, self.hidden_size)
            #self.anttion_p=tf.nn.softmax(self.anttion_p)
            # print('anttion_p###############')
            # print(self.anttion_p.get_shape())

    def _mul_attention(self):
        with tf.variable_scope('mul_attion'):
            mul_W = tf.get_variable(name="mul_attn_W",
                                    shape=[self.batch_size, 2 * self.hidden_size, 2 * self.hidden_size],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32)
            mul_V = tf.get_variable(name="mul_attn_V", shape=[self.batch_size, 2 * self.hidden_size, 1],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32)
            mul_U = tf.get_variable(name="mul_attn_U",
                                    shape=[self.batch_size, 2 * self.hidden_size, 2 * self.hidden_size],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32)
            # mul_p = tf.get_variable(name="mul_p",
            #                         shape=[16, 400, 128],
            #                         initializer=tf.contrib.layers.xavier_initializer(),
            #                         dtype=tf.float32)

            # mul_p=self.anttion_p[0]
            # for i in range(self.batch_size*self.max_p_num-1,1):
            #     mul_p=tf.concat([mul_p,self.anttion_p[i]],axis=0)
            mul_p=tf.reshape(self.anttion_p,[self.batch_size,-1,128])

            #print(mul_p.get_shape())
            mul_shape = tf.shape(mul_p)
            mul_output = tf.reshape(mul_p, [-1, 2 * self.hidden_size])
            mul_atten_hidden = tf.tanh(
                (tf.add
                    (
                    tf.matmul(mul_p, mul_W),
                    tf.matmul(mul_p, mul_U))*tf.cast(1000,tf.float32)))
            # print('mul_atten_h###############')
            # print(mul_atten_hidden.get_shape())
            max_c=tf.reduce_max(tf.reshape(tf.matmul(mul_atten_hidden, mul_V), [-1, mul_shape[1], 1]))
            #mul_alpha = tf.nn.softmax(
                #tf.reshape(tf.matmul(mul_atten_hidden, mul_V), [-1, mul_shape[1], 1])-max_c, axis=1)
            mul_alpha = tf.nn.softmax(
                tf.reshape(tf.matmul(mul_atten_hidden, mul_V), [-1, mul_shape[1], 1])-max_c)
            #mul_alpha = tf.reshape(tf.matmul(mul_atten_hidden, mul_V), [-1, mul_shape[1], 1])
            self.mul_h=mul_atten_hidden
            self.max=max_c
            self.mul_a=mul_alpha
            mul_output = tf.reshape(mul_output, [-1, mul_shape[1], 2 * self.hidden_size])

            mul_C = tf.multiply(mul_alpha, mul_output)
            # print('mul_C###############')
            # print(mul_C.get_shape())
            mul_output_C = tf.concat([mul_output, mul_C], axis=-1)
            mul_output_C=tf.reshape(mul_output_C,[self.batch_size*self.max_p_num,-1,256])
            # print('mul_output_C###############')
            # print(mul_output_C.get_shape())

            self.mul_anttion_p, _ = rnn('bi-gru', mul_output_C, self.c_len, self.hidden_size)
            #self.mul_anttion_p=tf.nn.softmax(self.mul_anttion_p)
            # if self.use_dropout:
            #     self.mul_anttion_p = tf.nn.dropout(self.mul_anttion_p, self.dropout)
            # # print('mul_anttion_p###############')
            # print(self.mul_anttion_p.get_shape())

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        N, PL, QL, CL, d, dc, nh = self._params()
        if self.use_position_attn:
            start_logits = tf.squeeze(
                conv(self._attention(tf.concat([self.enc[1], self.enc[2]], axis=-1), name="attn1"), 1, bias=False,
                     name="start_pointer"), -1)
            end_logits = tf.squeeze(
                conv(self._attention(tf.concat([self.enc[1], self.enc[3]], axis=-1), name="attn2"), 1, bias=False,
                     name="end_pointer"), -1)
        else:
            start_logits = tf.squeeze(
                conv(self.mul_anttion_p, 1, bias=False, name="start_pointer"), -1)
            end_logits = tf.squeeze(
                conv(self.mul_anttion_p, 1, bias=False, name="end_pointer"), -1)

        start_logits=tf.reshape(start_logits,[N,-1])
        self.sl=start_logits
        end_logits = tf.reshape(end_logits, [N, -1])
        self.el=end_logits
        self.logits = [mask_logits(start_logits, mask=tf.reshape(self.c_mask, [N, -1])),
                       mask_logits(end_logits, mask=tf.reshape(self.c_mask, [N, -1]))]

        self.logits1, self.logits2 = [l for l in self.logits]

        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                          tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))

        outer = tf.matrix_band_part(outer, 0, self.max_a_len)
        self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
    def _compute_loss(self):
        """
        The loss function
        """

        def focal_loss(logits, labels, weights=None, alpha=0.25, gamma=2):
            logits = tf.nn.sigmoid(logits)
            zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
            pos_p_sub = array_ops.where(labels > zeros, labels - logits, zeros)
            neg_p_sub = array_ops.where(labels > zeros, zeros, logits)
            cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - logits, 1e-8, 1.0))
            return tf.reduce_sum(cross_ent, 1)

        start_label = tf.one_hot(self.start_label, tf.shape(self.logits1)[1], axis=1)
        end_label = tf.one_hot(self.end_label, tf.shape(self.logits2)[1], axis=1)

        if self.loss_type == 'cross_entropy':

            start_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits1, labels=start_label)
            #print('start_loss###')
            #
            # print(self.sess.run(start_loss))
            self.s=start_loss
            end_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits2, labels=end_label)
            #print('end_loss####')
            #print(self.sess.run(end_loss))
            self.e=end_loss
            self.loss = tf.reduce_mean(start_loss + end_loss)

        else:
            start_loss = focal_loss(tf.nn.softmax(self.logits1, -1), start_label)
            end_loss = focal_loss(tf.nn.softmax(self.logits2, -1), end_label)
            self.loss = tf.reduce_mean(start_loss + end_loss)
        self.logger.info("loss type %s" % self.loss_type)

        self.all_params = tf.trainable_variables()

        if self.l2_norm is not None:
            self.logger.info("applying l2 loss")
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            #print('l2_loss#####')
            #print(self.sess.run(l2_loss))
            self.loss += l2_loss

        if self.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(self.config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.shadow_vars = []
                self.global_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.shadow_vars.append(v)
                        self.global_vars.append(var)
                self.assign_vars = []
                for g, v in zip(self.global_vars, self.shadow_vars):
                    self.assign_vars.append(tf.assign(g, v))


    def _create_train_op(self):
        #self.lr = tf.minimum(self.learning_rate, self.learning_rate / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self.lr = self.learning_rate

        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optim_type == 'adamW':
            self.optimizer = AdamWOptimizer(self.weight_decay,
                                                            learning_rate=self.lr)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        
        self.logger.info("applying optimize %s" % self.optim_type)
        trainable_vars = tf.trainable_variables()
        if self.clip_weight:
            # clip_weight
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.max_norm_grad)
            grad_var_pairs = zip(grads, tvars)
            self.train_op = self.optimizer.apply_gradients(grad_var_pairs, name='apply_grad')
        else:
            self.train_op = self.optimizer.minimize(self.loss)

    def _attention(self, output, name='attn', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            W = tf.get_variable(name="attn_W",
                                shape=[2 * self.config.hidden_size, 2 * self.config.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                # initializer=tf.truncated_normal_initializer(),
                                # initializer=tf.keras.initializers.lecun_normal(),
                                dtype=tf.float32)
            V = tf.get_variable(name="attn_V", shape=[2 * self.config.hidden_size, 1],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                # initializer=tf.truncated_normal_initializer(),
                                # initializer=tf.keras.initializers.lecun_normal(),
                                dtype=tf.float32)
            U = tf.get_variable(name="attn_U",
                                shape=[2 * self.config.hidden_size, 2 * self.config.hidden_size],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                # initializer=tf.truncated_normal_initializer(),
                                # initializer=tf.keras.initializers.lecun_normal(),
                                dtype=tf.float32)

            self.position_emb = tf.reshape(self.position_emb, [-1, 2 * self.config.hidden_size])
            shape = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size])

            atten_hidden = tf.tanh(
                tf.add(
                    tf.matmul(self.position_emb, W),
                    tf.matmul(output, U)))
            alpha = tf.nn.softmax(
                tf.reshape(tf.matmul(atten_hidden, V), [-1, shape[1], 1]), axis=1)
            output = tf.reshape(output, [-1, shape[1], 2 * self.config.hidden_size])
            C = tf.multiply(alpha, output)
            return tf.concat([output, C], axis=-1)



    def _train_epoch(self, train_batches, dropout):
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 1000, 0
        for bitx, batch in enumerate(train_batches, 1):
            if len(batch['passage_token_ids']) < self.batch_size or len(
                    batch['question_token_ids']) < self.batch_size or len(batch['start_id']) < self.batch_size or len(
                    batch['end_id']) < self.batch_size:
                continue

            feed_dict = {self.c: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.qh: batch['question_char_ids'],
                         self.ch: batch["passage_char_ids"],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout: dropout}
            #print('c_LEN######')
            #print(batch['raw_data'])
            #print(self.c.get_shape())
            #print('q_LEN######')
            #print(self.q.get_shape())
            #print(feed_dict)
            #try:

            _, loss, global_step = self.sess.run([self.train_op, self.loss, self.global_step], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            #print(total_num)
            n_batch_loss += loss
            #except Exception as e:
                #print("Error>>>", e)

            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        #print("total_num", total_num)
        return 1.0 * total_loss / total_num

    def _params(self):
    	return (self.batch_size if not self.demo else 1, self.max_p_len,
    		self.max_q_len, self.max_ch_len, self.hidden_size,
                self.char_embed_size, self.head_size)

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=0.5, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_word_id(self.vocab.pad_token)
        pad_char_id = self.vocab.get_char_id(self.vocab.pad_token)
        max_rouge_l = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))

            train_batches = data.next_batch('train', batch_size, pad_id, pad_char_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.next_batch('dev', batch_size, pad_id, pad_char_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Rouge-L'] > max_rouge_l:
                        self.save(save_dir, save_prefix)
                        max_rouge_l = bleu_rouge['Rouge-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))


    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        # print('eval_batches######')
        #
        # print(eval_batches.get_shape())
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0

        def mul_dict(data):
            a = []
            i = 1
            a_one = []
            for line in data:
                a_one.append(line)
                if i % self.max_p_num == 0:
                    a.append(a_one)
                    a_one = []
                i += 1
            return a
        for b_itx, batch in enumerate(eval_batches):
            feed_dict =  {self.c: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.qh: batch['question_char_ids'],
                         self.ch: batch["passage_char_ids"],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout:0.0}
            # mul_dict={self.c:mul_dict( batch['passage_token_ids']),
            #              self.q: mul_dict(batch['question_token_ids']),
            #              self.qh: mul_dict(batch['question_char_ids']),
            #              self.ch: mul_dict(batch["passage_char_ids"]),
            #              self.start_label: mul_dict(batch['start_id']),
            #              self.end_label: mul_dict(batch['end_id']),
            #              self.dropout:0.0}
            # mul_a,max=self.sess.run([self.mul_a,self.max],feed_dict)
            # print('mul_a#####')
            # print(mul_a,max)

            try:
                f,mp,sq,ssc,sc,c_emb,q_emb,match,fuse,sa,sl,el,s,e,start_probs, end_probs, loss = self.sess.run([self.f,self.mp,self.sep_q_encodes,self.sc,self.sep_c_encodes,self.c_emb,self.q_emb,self.match_p_encodes,self.fuse_p_encodes,self.anttion_p,self.sl,self.el,self.s,self.e,self.logits1,
                                                              self.logits2, self.loss], feed_dict)


                total_loss += loss * len(batch['raw_data'])
                total_num += len(batch['raw_data'])

                padded_p_len = len(batch['passage_token_ids'][0])
                for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                    best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                    if save_full_info:
                        sample['pred_answers'] = [best_answer]
                        pred_answers.append(sample)
                    else:
                        pred_answers.append({'question_id': sample['question_id'],
                                             'question_type': sample['question_type'],
                                             'answers': [best_answer],
                                             'entity_answers': [[]],
                                             'yesno_answers': []})
                    if 'answers' in sample:
                        ref_answers.append({'question_id': sample['question_id'],
                                            'question_type': sample['question_type'],
                                            'answers': sample['answers'],
                                            'entity_answers': [[]],
                                            'yesno_answers': []})
            except:
                continue


        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

            # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None



        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """ 90
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))

