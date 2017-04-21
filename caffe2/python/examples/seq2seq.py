## @package seq2seq
# Module caffe2.python.examples.seq2seq
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import logging
import math
import numpy as np
import random
import time
import sys
from timeit import default_timer as timer

from itertools import izip

import caffe2.proto.caffe2_pb2 as caffe2_pb2
from caffe2.python import core, workspace, recurrent, data_parallel_model
from caffe2.python.examples import seq2seq_util

import seq2seq_data

import matplotlib.pyplot as plt
reload(sys)
sys.setdefaultencoding('utf8')

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stderr))

Batch = collections.namedtuple('Batch', [
    'encoder_inputs',
    'encoder_lengths',
    'decoder_inputs',
    'decoder_lengths',
    'targets',
    'target_weights',
])

_PAD_ID = 0
_GO_ID = 1
_EOS_ID = 2
EOS = '<EOS>'
UNK = '<UNK>'
GO = '<GO>'
PAD = '<PAD>'


def prepare_batch(batch):
    (source_inputs, source_lengths,
     target_inputs, target_lengths) = batch

    # encoder_inputs = reverse(source_inputs)
    encoder_inputs = np.full(source_inputs.shape, _PAD_ID,
                             dtype=source_inputs.dtype)
    for i, (row, length) in enumerate(izip(source_inputs, source_lengths)):
        encoder_inputs[i, :length] = row[length - 1::-1]
    encoder_lengths = source_lengths

    # decoder_inputs = [_GO_ID] + target_inputs
    decoder_inputs = np.hstack([
        np.full((len(target_inputs), 1), _GO_ID, dtype=target_inputs.dtype),
        target_inputs])
    decoder_lengths = target_lengths + 1

    # targets = target_inputs + [_EOS_ID]
    targets = np.hstack([
        target_inputs,
        np.full((len(target_inputs), 1), _PAD_ID, dtype=target_inputs.dtype)])
    target_weights = np.zeros(targets.shape, dtype=np.float32)
    for i, length in enumerate(target_lengths):
        targets[i, length] = _EOS_ID
        target_weights[i, :length] = 1

    return Batch(
        encoder_inputs=encoder_inputs.transpose(),
        encoder_lengths=encoder_lengths,
        decoder_inputs=decoder_inputs.transpose(),
        decoder_lengths=decoder_lengths,
        targets=targets.transpose(),
        target_weights=target_weights.transpose(),
    )


class Seq2SeqModelCaffe2:

    def _build_model(
        self,
        init_params,
    ):
        model = seq2seq_util.ModelHelper(init_params=init_params)
        self._build_shared(model)
        self._build_embeddings(model)

        forward_model = seq2seq_util.ModelHelper(init_params=init_params)
        self._build_shared(forward_model)
        self._build_embeddings(forward_model)

        if self.num_gpus == 0:
            loss_blobs = self.model_build_fun(model)
            model.AddGradientOperators(loss_blobs)
            self.norm_clipped_grad_update(
                model,
                scope='norm_clipped_grad_update'
            )
            self.forward_model_build_fun(forward_model)

        else:
            assert (self.batch_size % self.num_gpus) == 0

            data_parallel_model.Parallelize_GPU(
                forward_model,
                input_builder_fun=lambda m: None,
                forward_pass_builder_fun=self.forward_model_build_fun,
                param_update_builder_fun=None,
                devices=range(self.num_gpus),
            )

            def clipped_grad_update_bound(model):
                self.norm_clipped_grad_update(
                    model,
                    scope='norm_clipped_grad_update',
                )

            data_parallel_model.Parallelize_GPU(
                model,
                input_builder_fun=lambda m: None,
                forward_pass_builder_fun=self.model_build_fun,
                param_update_builder_fun=clipped_grad_update_bound,
                devices=range(self.num_gpus),
            )
        self.norm_clipped_sparse_grad_update(
            model,
            scope='norm_clipped_sparse_grad_update',
        )
        self.model = model
        self.forward_net = forward_model.net

    def _build_embedding_encoder(
        self,
        model,
        inputs,
        input_lengths,
        vocab_size,
        embeddings,
        embedding_size,
        use_attention,
        num_gpus,
        forward_only=False,
    ):
        if num_gpus == 0:
            embedded_encoder_inputs = model.net.Gather(
                [embeddings, inputs],
                ['embedded_encoder_inputs'],
            )
        else:
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                embedded_encoder_inputs_cpu = model.net.Gather(
                    [embeddings, inputs],
                    ['embedded_encoder_inputs_cpu'],
                )
            embedded_encoder_inputs = model.CopyCPUToGPU(
                embedded_encoder_inputs_cpu,
                'embedded_encoder_inputs',
            )

        if self.encoder_type == 'rnn':
            assert len(self.encoder_params['encoder_layer_configs']) == 1
            encoder_num_units = (
                self.encoder_params['encoder_layer_configs'][0]['num_units']
            )
            encoder_initial_cell_state = model.param_init_net.ConstantFill(
                [],
                ['encoder_initial_cell_state'],
                shape=[encoder_num_units],
                value=0.0,
            )
            encoder_initial_hidden_state = (
                model.param_init_net.ConstantFill(
                    [],
                    'encoder_initial_hidden_state',
                    shape=[encoder_num_units],
                    value=0.0,
                )
            )
            # Choose corresponding rnn encoder function
            if self.encoder_params['use_bidirectional_encoder']:
                rnn_encoder_func = seq2seq_util.rnn_bidirectional_encoder
                encoder_output_dim = 2 * encoder_num_units
            else:
                rnn_encoder_func = seq2seq_util.rnn_unidirectional_encoder
                encoder_output_dim = encoder_num_units

            (
                encoder_outputs,
                final_encoder_hidden_state,
                final_encoder_cell_state,
            ) = rnn_encoder_func(
                model,
                embedded_encoder_inputs,
                input_lengths,
                encoder_initial_hidden_state,
                encoder_initial_cell_state,
                embedding_size,
                encoder_num_units,
                use_attention,
            )
            weighted_encoder_outputs = None
        else:
            raise ValueError('Unsupported encoder type {}'.format(
                self.encoder_type))

        return (
            encoder_outputs,
            weighted_encoder_outputs,
            final_encoder_hidden_state,
            final_encoder_cell_state,
            encoder_output_dim,
        )

    def output_projection(
        self,
        model,
        decoder_outputs,
        decoder_output_size,
        target_vocab_size,
        decoder_softmax_size,
    ):
        if decoder_softmax_size is not None:
            decoder_outputs = model.FC(
                decoder_outputs,
                'decoder_outputs_scaled',
                dim_in=decoder_output_size,
                dim_out=decoder_softmax_size,
            )
            decoder_output_size = decoder_softmax_size

        output_projection_w = model.param_init_net.XavierFill(
            [],
            'output_projection_w',
            shape=[self.target_vocab_size, decoder_output_size],
        )

        output_projection_b = model.param_init_net.XavierFill(
            [],
            'output_projection_b',
            shape=[self.target_vocab_size],
        )
        model.params.extend([
            output_projection_w,
            output_projection_b,
        ])
        output_logits = model.net.FC(
            [
                decoder_outputs,
                output_projection_w,
                output_projection_b,
            ],
            ['output_logits'],
        )
        return output_logits

    def _build_shared(self, model):
        optimizer_params = self.model_params['optimizer_params']
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            self.learning_rate = model.AddParam(
                name='learning_rate',
                init_value=float(optimizer_params['learning_rate']),
                trainable=False,
            )
            self.global_step = model.AddParam(
                name='global_step',
                init_value=0,
                trainable=False,
            )
            self.start_time = model.AddParam(
                name='start_time',
                init_value=time.time(),
                trainable=False,
            )

    def _build_embeddings(self, model):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            sqrt3 = math.sqrt(3)
            self.encoder_embeddings = model.param_init_net.UniformFill(
                [],
                'encoder_embeddings',
                shape=[
                    self.source_vocab_size,
                    self.model_params['encoder_embedding_size'],
                ],
                min=-sqrt3,
                max=sqrt3,
            )
            model.params.append(self.encoder_embeddings)
            self.decoder_embeddings = model.param_init_net.UniformFill(
                [],
                'decoder_embeddings',
                shape=[
                    self.target_vocab_size,
                    self.model_params['decoder_embedding_size'],
                ],
                min=-sqrt3,
                max=sqrt3,
            )
            model.params.append(self.decoder_embeddings)

    def model_build_fun(self, model, forward_only=False, loss_scale=None):
        encoder_inputs = model.net.AddExternalInput(
            workspace.GetNameScope() + 'encoder_inputs',
        )
        encoder_lengths = model.net.AddExternalInput(
            workspace.GetNameScope() + 'encoder_lengths',
        )
        decoder_inputs = model.net.AddExternalInput(
            workspace.GetNameScope() + 'decoder_inputs',
        )
        decoder_lengths = model.net.AddExternalInput(
            workspace.GetNameScope() + 'decoder_lengths',
        )
        targets = model.net.AddExternalInput(
            workspace.GetNameScope() + 'targets',
        )
        target_weights = model.net.AddExternalInput(
            workspace.GetNameScope() + 'target_weights',
        )
        attention_type = self.model_params['attention']
        assert attention_type in ['none', 'regular']

        (
            encoder_outputs,
            weighted_encoder_outputs,
            final_encoder_hidden_state,
            final_encoder_cell_state,
            encoder_output_dim,
        ) = self._build_embedding_encoder(
            model=model,
            inputs=encoder_inputs,
            input_lengths=encoder_lengths,
            vocab_size=self.source_vocab_size,
            embeddings=self.encoder_embeddings,
            embedding_size=self.model_params['encoder_embedding_size'],
            use_attention=(attention_type != 'none'),
            num_gpus=self.num_gpus,
            forward_only=forward_only,
        )

        assert len(self.model_params['decoder_layer_configs']) == 1
        decoder_num_units = (
            self.model_params['decoder_layer_configs'][0]['num_units']
        )

        if attention_type == 'none':
            decoder_initial_hidden_state = model.FC(
                final_encoder_hidden_state,
                'decoder_initial_hidden_state',
                encoder_output_dim,
                decoder_num_units,
                axis=2,
            )
            decoder_initial_cell_state = model.FC(
                final_encoder_cell_state,
                'decoder_initial_cell_state',
                encoder_output_dim,
                decoder_num_units,
                axis=2,
            )
        else:
            decoder_initial_hidden_state = model.param_init_net.ConstantFill(
                [],
                'decoder_initial_hidden_state',
                shape=[decoder_num_units],
                value=0.0,
            )
            decoder_initial_cell_state = model.param_init_net.ConstantFill(
                [],
                'decoder_initial_cell_state',
                shape=[decoder_num_units],
                value=0.0,
            )
            initial_attention_weighted_encoder_context = (
                model.param_init_net.ConstantFill(
                    [],
                    'initial_attention_weighted_encoder_context',
                    shape=[encoder_output_dim],
                    value=0.0,
                )
            )

        if self.num_gpus == 0:
            embedded_decoder_inputs = model.net.Gather(
                [self.decoder_embeddings, decoder_inputs],
                ['embedded_decoder_inputs'],
            )
        else:
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                embedded_decoder_inputs_cpu = model.net.Gather(
                    [self.decoder_embeddings, decoder_inputs],
                    ['embedded_decoder_inputs_cpu'],
                )
            embedded_decoder_inputs = model.CopyCPUToGPU(
                embedded_decoder_inputs_cpu,
                'embedded_decoder_inputs',
            )

        # seq_len x batch_size x decoder_embedding_size
        if attention_type == 'none':
            decoder_outputs, _, _, _ = recurrent.LSTM(
                model=model,
                input_blob=embedded_decoder_inputs,
                seq_lengths=decoder_lengths,
                initial_states=(
                    decoder_initial_hidden_state,
                    decoder_initial_cell_state,
                ),
                dim_in=self.model_params['decoder_embedding_size'],
                dim_out=decoder_num_units,
                scope='decoder',
                outputs_with_grads=[0],
            )
            decoder_output_size = decoder_num_units
        else:
            (
                decoder_outputs, _, _, _,
                attention_weighted_encoder_contexts, _
            ) = recurrent.LSTMWithAttention(
                model=model,
                decoder_inputs=embedded_decoder_inputs,
                decoder_input_lengths=decoder_lengths,
                initial_decoder_hidden_state=decoder_initial_hidden_state,
                initial_decoder_cell_state=decoder_initial_cell_state,
                initial_attention_weighted_encoder_context=(
                    initial_attention_weighted_encoder_context
                ),
                encoder_output_dim=encoder_output_dim,
                encoder_outputs=encoder_outputs,
                decoder_input_dim=self.model_params['decoder_embedding_size'],
                decoder_state_dim=decoder_num_units,
                scope='decoder',
                outputs_with_grads=[0, 4],
            )
            decoder_outputs, _ = model.net.Concat(
                [decoder_outputs, attention_weighted_encoder_contexts],
                [
                    'states_and_context_combination',
                    '_states_and_context_combination_concat_dims',
                ],
                axis=2,
            )
            decoder_output_size = decoder_num_units + encoder_output_dim

        # we do softmax over the whole sequence
        # (max_length in the batch * batch_size) x decoder embedding size
        # -1 because we don't know max_length yet
        decoder_outputs_flattened, _ = model.net.Reshape(
            [decoder_outputs],
            [
                'decoder_outputs_flattened',
                'decoder_outputs_and_contexts_combination_old_shape',
            ],
            shape=[-1, decoder_output_size],
        )
        output_logits = self.output_projection(
            model=model,
            decoder_outputs=decoder_outputs_flattened,
            decoder_output_size=decoder_output_size,
            target_vocab_size=self.target_vocab_size,
            decoder_softmax_size=self.model_params['decoder_softmax_size'],
        )
        targets, _ = model.net.Reshape(
            [targets],
            ['targets', 'targets_old_shape'],
            shape=[-1],
        )
        target_weights, _ = model.net.Reshape(
            [target_weights],
            ['target_weights', 'target_weights_old_shape'],
            shape=[-1],
        )
        output_probs = model.net.Softmax(
            [output_logits],
            ['output_probs'],
            engine=('CUDNN' if self.num_gpus > 0 else None),
        )
        label_cross_entropy = model.net.LabelCrossEntropy(
            [output_probs, targets],
            ['label_cross_entropy'],
        )
        weighted_label_cross_entropy = model.net.Mul(
            [label_cross_entropy, target_weights],
            'weighted_label_cross_entropy',
        )
        total_loss_scalar = model.net.SumElements(
            [weighted_label_cross_entropy],
            'total_loss_scalar',
        )
        total_loss_scalar_weighted = model.net.Scale(
            [total_loss_scalar],
            'total_loss_scalar_weighted',
            scale=1.0 / self.batch_size,
        )
        return [total_loss_scalar_weighted]

    def forward_model_build_fun(self, model, loss_scale=None):
        return self.model_build_fun(
            model=model,
            forward_only=True,
            loss_scale=loss_scale
        )

    def _calc_norm_ratio(self, model, params, scope, ONE):
        with core.NameScope(scope):
            grad_squared_sums = []
            for i, param in enumerate(params):
                logger.info(param)
                grad = (
                    model.param_to_grad[param]
                    if not isinstance(
                        model.param_to_grad[param],
                        core.GradientSlice,
                    ) else model.param_to_grad[param].values
                )
                grad_squared = model.net.Sqr(
                    [grad],
                    'grad_{}_squared'.format(i),
                )
                grad_squared_sum = model.net.SumElements(
                    grad_squared,
                    'grad_{}_squared_sum'.format(i),
                )
                grad_squared_sums.append(grad_squared_sum)

            grad_squared_full_sum = model.net.Sum(
                grad_squared_sums,
                'grad_squared_full_sum',
            )
            global_norm = model.net.Pow(
                grad_squared_full_sum,
                'global_norm',
                exponent=0.5,
            )
            clip_norm = model.param_init_net.ConstantFill(
                [],
                'clip_norm',
                shape=[],
                value=float(self.model_params['max_gradient_norm']),
            )
            max_norm = model.net.Max(
                [global_norm, clip_norm],
                'max_norm',
            )
            norm_ratio = model.net.Div(
                [clip_norm, max_norm],
                'norm_ratio',
            )
            return norm_ratio

    def _apply_norm_ratio(
        self, norm_ratio, model, params, learning_rate, scope, ONE
    ):
        with core.NameScope(scope):
            update_coeff = model.net.Mul(
                [learning_rate, norm_ratio],
                'update_coeff',
                broadcast=1,
            )
        if(self.optimizer == "adagrad"):
            neg_update_coeff = model.net.Negative(
                [update_coeff],
                'neg_update_coeff',
            )
            for param in params:
                param_grad = model.param_to_grad[param]
                grad_history = model.param_init_net.ConstantFill(
                    [param], param + '_history', value=0.0)

                if isinstance(param_grad, core.GradientSlice):
                    param_grad_values = param_grad.values
                    param_grad_indices = param_grad.indices

                    model.net.SparseAdagrad(
                        [param, grad_history, param_grad_indices, param_grad_values, neg_update_coeff],
                        [param, grad_history],
                        epsilon=1.0)
                else:
                    model.net.Adagrad(
                        [param, grad_history, param_grad, neg_update_coeff],
                        [param, grad_history],
                        epsilon=1.0)
        elif(self.optimizer == "adam"):
            neg_update_coeff = model.net.Negative(
                [update_coeff],
                'neg_update_coeff',
            )
            for param in params:
                param_grad = model.param_to_grad[param]
                param_momentum1 = model.param_init_net.ConstantFill(
                    [param], param + '_momentum1', value=0.0)
                param_momentum2 = model.param_init_net.ConstantFill(
                    [param], param + '_momentum2', value=0.0)
                if isinstance(param_grad, core.GradientSlice):
                    param_grad_values = param_grad.values
                    param_grad_indices = param_grad.indices
                    model.net.SparseAdam(
                        [param, param_momentum1, param_momentum2,
                          param_grad_indices, param_grad_values, neg_update_coeff, self.global_step],
                        [param, param_momentum1, param_momentum2],
                        beta1=0.999,
                        beta2=0.00000001,
                        epsilon=1.0)
                else:
                    model.net.Adam(
                        [param, param_momentum1, param_momentum2,
                          param_grad, neg_update_coeff, self.global_step],
                        [param, param_momentum1, param_momentum2],
                        beta1=0.999,
                        beta2=0.00000001,
                        epsilon=1.0)
        elif(self.optimizer == "momentum"):
            for param in params:
                param_grad = model.param_to_grad[param]
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0)
                if isinstance(param_grad, core.GradientSlice):
                    param_grad_values = param_grad.values
                    param_grad_indices = param_grad.indices

                    model.net.SparseMomentumSGDUpdate(
                        [param_grad_values, param_momentum, update_coeff, param, param_grad_indices],
                        [param_grad_values, param_momentum, param],
                        nesterov=False, momentum=0.9)
                else:
                    model.net.MomentumSGDUpdate(
                        [param_grad, param_momentum, update_coeff, param],
                        [param_grad, param_momentum, param],
                        nesterov=False, momentum=0.9)
        elif(self.optimizer == "sgd"):
            neg_update_coeff = model.net.Negative(
                [update_coeff],
                'neg_update_coeff',
            )
            for param in params:
                param_grad = model.param_to_grad[param]
                if isinstance(param_grad, core.GradientSlice):
                    param_grad_values = param_grad.values
                    model.net.ScatterWeightedSum(
                        [
                            param,
                            ONE,
                            param_grad.indices,
                            param_grad_values,
                            neg_update_coeff,
                        ],
                        param,
                    )
                else:
                    model.net.WeightedSum(
                        [
                            param,
                            ONE,
                            param_grad,
                            neg_update_coeff,
                        ],
                        param,
                    )
        else:
            raise ValueError('Unsupported optimizer type {}'.format(
                self.optimizer))

    def norm_clipped_grad_update(self, model, scope):

        if self.num_gpus == 0:
            learning_rate = self.learning_rate
        else:
            learning_rate = model.CopyCPUToGPU(self.learning_rate, 'LR')

        params = []
        for param in model.GetParams(top_scope=True):
            if param in model.param_to_grad:
                if not isinstance(
                    model.param_to_grad[param],
                    core.GradientSlice,
                ):
                    params.append(param)

        ONE = model.param_init_net.ConstantFill(
            [],
            'ONE',
            shape=[1],
            value=1.0,
        )
        logger.info('Dense trainable variables: ')
        norm_ratio = self._calc_norm_ratio(model, params, scope, ONE)
        self._apply_norm_ratio(
            norm_ratio, model, params, learning_rate, scope, ONE
        )

    def norm_clipped_sparse_grad_update(self, model, scope):
        learning_rate = self.learning_rate

        params = []
        for param in model.GetParams(top_scope=True):
            if param in model.param_to_grad:
                if isinstance(
                    model.param_to_grad[param],
                    core.GradientSlice,
                ):
                    params.append(param)

        ONE = model.param_init_net.ConstantFill(
            [],
            'ONE',
            shape=[1],
            value=1.0,
        )
        logger.info('Sparse trainable variables: ')
        norm_ratio = self._calc_norm_ratio(model, params, scope, ONE)
        self._apply_norm_ratio(
            norm_ratio, model, params, learning_rate, scope, ONE
        )

    def total_loss_scalar(self):
        if self.num_gpus == 0:
            return workspace.FetchBlob('total_loss_scalar')
        else:
            total_loss = 0
            for i in range(self.num_gpus):
                name = 'gpu_{}/total_loss_scalar'.format(i)
                gpu_loss = workspace.FetchBlob(name)
                total_loss += gpu_loss
            return total_loss

    def _init_model(self):
        workspace.RunNetOnce(self.model.param_init_net)

        def create_net(net):
            workspace.CreateNet(
                net,
                input_blobs=map(str, net.external_inputs),
            )

        create_net(self.model.net)
        create_net(self.forward_net)

    def __init__(
        self,
        model_params,
        source_vocab_size,
        target_vocab_size,
        num_gpus=1,
        num_cpus=1,
        optimizer="sgd",
    ):
        self.model_params = model_params
        self.encoder_type = 'rnn'
        self.encoder_params = model_params['encoder_type']
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.optimizer= optimizer
        self.batch_size = model_params['batch_size']

        workspace.GlobalInit([
            'caffe2',
            # NOTE: modify log level for debugging purposes
            '--caffe2_log_level=0',
            # NOTE: modify log level for debugging purposes
            '--v=0',
            # Fail gracefully if one of the threads fails
            '--caffe2_handle_executor_threads_exceptions=1',
            '--caffe2_mkl_num_threads=' + str(self.num_cpus),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        workspace.ResetWorkspace()

    def initialize_from_scratch(self):
        logger.info('Initializing Seq2SeqModelCaffe2 from scratch: Start')
        self._build_model(init_params=True)
        self._init_model()
        logger.info('Initializing Seq2SeqModelCaffe2 from scratch: Finish')

    def get_current_step(self):
        return workspace.FetchBlob(self.global_step)[0]

    def inc_current_step(self):
        workspace.FeedBlob(
            self.global_step,
            np.array([self.get_current_step() + 1]),
        )

    def step(
        self,
        batch,
        forward_only
    ):
        if self.num_gpus < 1:
            batch_obj = prepare_batch(batch)
            for batch_obj_name, batch_obj_value in izip(
                Batch._fields,
                batch_obj,
            ):
                workspace.FeedBlob(batch_obj_name, batch_obj_value)
        else:
            for i in range(self.num_gpus):
                gpu_batch = batch[i::self.num_gpus]
                batch_obj = prepare_batch(gpu_batch)
                for batch_obj_name, batch_obj_value in izip(
                    Batch._fields,
                    batch_obj,
                ):
                    name = 'gpu_{}/{}'.format(i, batch_obj_name)
                    if batch_obj_name in ['encoder_inputs', 'decoder_inputs']:
                        dev = core.DeviceOption(caffe2_pb2.CPU)
                    else:
                        dev = core.DeviceOption(caffe2_pb2.CUDA, i)
                    workspace.FeedBlob(name, batch_obj_value, device_option=dev)

        if forward_only:
            workspace.RunNet(self.forward_net)
        else:
            workspace.RunNet(self.model.net)
            self.inc_current_step()

        return self.total_loss_scalar()


def run_seq2seq_model(args, model_params=None):
    (source_vocab, target_vocab,
     train_data, test_data) = seq2seq_data.get_data(args)

    with Seq2SeqModelCaffe2(
        model_params=model_params,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        num_gpus=args.num_gpus,
        num_cpus=20,
        optimizer=args.optimizer,
    ) as model_obj:
        model_obj.initialize_from_scratch()

        with open("seq2seq.pbtxt", "w") as fid:
            fid.write(str(model_obj.model.net.Proto()))
        with open("seq2seq_init.pbtxt", "w") as fid:
            fid.write(str(model_obj.model.param_init_net.Proto()))
        with open("seq2seq_forward.pbtxt", "w") as fid:
            fid.write(str(model_obj.forward_net.Proto()))

        sequences_per_epoch = sum([len(train_data[key][0])
                                   for key in train_data])
        iterations_per_epoch = int(np.ceil(sequences_per_epoch / args.batch_size))
        tokens_per_epoch = sum([np.sum(train_data[key][1])
                                for key in train_data])
        tokens_with_padding_per_epoch = sum([train_data[key][0].size
                                             for key in train_data])
        display_interval = int(np.ceil(iterations_per_epoch / 100.0))
        test_interval = int(np.ceil(iterations_per_epoch / 10.0))

        print("Iterations per epoch:    ", iterations_per_epoch)
        print("Sequences per epoch:     ", sequences_per_epoch)
        print("Tokens per epoch:        ", tokens_per_epoch)
        print("Tokens+padding per epoch:", tokens_with_padding_per_epoch)
        print("display_interval:        ", display_interval)
        print("test_interval:           ", test_interval)

        for epoch in range(args.epochs):
            # For display
            epoch_start = timer()
            current_iter = 0
            last_display_iter = 0
            display_time = 0
            display_loss = 0
            display_sequences = 0
            display_tokens = 0
            display_padding = 0

            for batch in seq2seq_data.iterate_epoch(
                    train_data, args.batch_size, shuffle=True):
                current_iter += 1
                encoder_token_num = np.sum(batch[1])
                decoder_token_num = np.sum(batch[3])

                start_time = timer()
                loss = model_obj.step(batch=batch, forward_only=False)
                step_time = timer() - start_time

                # Updates for display
                display_time += step_time
                display_loss += loss
                step_tokens = np.sum(batch[1])
                display_sequences += len(batch[0])
                display_tokens += step_tokens
                display_padding += (batch[0].size - step_tokens)
                current_epoch = epoch + float(current_iter) / iterations_per_epoch

                # Display
                if current_iter % display_interval == 0:
                    perplexity = pow(2, display_loss / display_tokens)
                    print("Epoch %f/%d" % (current_epoch, args.epochs))
                    print("    Displaying after %d iterations, %f seconds" %
                          (current_iter - last_display_iter, display_time))
                    print("    Training loss=%f, perplexity=%f" %
                          (loss, perplexity))
                    print("    Iterations/second:     %f" %
                          ((current_iter - last_display_iter) / display_time,))
                    print("    Sequences/second:      %f" %
                          (display_sequences / display_time,))
                    print("    Tokens/second:         %f" %
                          (display_tokens / display_time,))
                    print("    Tokens+padding/second: %f" %
                          ((display_tokens + display_padding) / display_time,))
                    last_display_iter = current_iter
                    display_time = 0
                    display_loss = 0
                    display_sequences = 0
                    display_tokens = 0
                    display_padding = 0

                if current_iter % test_interval == 0:
                    print("\nEvaluating model ...")
                    eval_start = timer()
                    eval_iterations = 0
                    eval_loss = 0
                    eval_sequences = 0
                    eval_tokens = 0
                    eval_padding = 0
                    for batch in seq2seq_data.iterate_epoch(
                            test_data, args.batch_size):
                        loss = model_obj.step(batch=batch, forward_only=True)
                        eval_iterations += 1
                        eval_loss += loss
                        eval_sequences += len(batch[0])
                        step_tokens = np.sum(batch[1])
                        eval_tokens += step_tokens
                        eval_padding += (batch[0].size - step_tokens)

                    eval_time = timer() - eval_start

                    perplexity = pow(2, eval_loss / eval_tokens)
                    print("    Displaying after %d iterations, %f seconds" %
                          (eval_iterations, eval_time))
                    print("    Evaluation loss=%f, perplexity=%f" %
                          (eval_loss, perplexity))
                    print("    Iterations/second:     %f" %
                          (eval_iterations / eval_time,))
                    print("    Sequences/second:      %f" %
                          (eval_sequences / eval_time,))
                    print("    Tokens/second:         %f" %
                          (eval_tokens / eval_time,))
                    print("    Tokens+padding/second: %f" %
                          ((eval_tokens + eval_padding) / eval_time,))
                    print()

            epoch_time = timer() - epoch_start
            print("\nEpoch %d finished in %d seconds.\n" %
                  (epoch + 1, int(round(epoch_time))))

def run_seq2seq_rnn_unidirection_with_no_attention(args):
    run_seq2seq_model(args, model_params=dict(
        attention=('regular' if args.use_attention else 'none'),
        decoder_layer_configs=[
            dict(
                num_units=args.decoder_cell_num_units,
            ),
        ],
        encoder_type=dict(
            encoder_layer_configs=[
                dict(
                    num_units=args.encoder_cell_num_units,
                ),
            ],
            use_bidirectional_encoder=args.use_bidirectional_encoder,
        ),
        batch_size=args.batch_size,
        optimizer_params=dict(
            learning_rate=args.learning_rate,
        ),
        encoder_embedding_size=args.encoder_embedding_size,
        decoder_embedding_size=args.decoder_embedding_size,
        decoder_softmax_size=args.decoder_softmax_size,
        max_gradient_norm=args.max_gradient_norm,
    ))


def main():
    random.seed(31415)
    parser = argparse.ArgumentParser(
        description='Caffe2: Seq2Seq Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    seq2seq_data.addParserArguments(parser)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of iterations over training data')
    parser.add_argument('--learning-rate', type=float, default=0.5,
                        help='Learning rate')
    parser.add_argument('--max-gradient-norm', type=float, default=1.0,
                        help='Max global norm of gradients at the end of each '
                        'backward pass. We do clipping to match the number.')
    parser.add_argument('--use-bidirectional-encoder', action='store_true',
                        help='Set flag to use bidirectional recurrent network '
                        'in encoder')
    parser.add_argument('--use-attention', action='store_true',
                        help='Set flag to use seq2seq with attention model')
    parser.add_argument('--encoder-cell-num-units', type=int, default=256,
                        help='Number of cell units in the encoder layer')
    parser.add_argument('--decoder-cell-num-units', type=int, default=512,
                        help='Number of cell units in the decoder layer')
    parser.add_argument('--encoder-embedding-size', type=int, default=256,
                        help='Size of embedding in the encoder layer')
    parser.add_argument('--decoder-embedding-size', type=int, default=512,
                        help='Size of embedding in the decoder layer')
    parser.add_argument('--decoder-softmax-size', type=int, default=None,
                        help='Size of softmax layer in the decoder')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='Number of GPUs for data parallel model')
    parser.add_argument('--optimizer', type=str,
                        help='Optimizer type: sgd, momentum, adagrad')

    args = parser.parse_args()

    run_seq2seq_rnn_unidirection_with_no_attention(args)


if __name__ == '__main__':
    main()
