## @package seq2seq_util
# Module caffe2.python.examples.seq2seq_util
""" A bunch of util functions to build Seq2Seq models with Caffe2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import recurrent
from caffe2.python.cnn import CNNModelHelper


class ModelHelper(CNNModelHelper):

    def __init__(self, init_params=True):
        super(ModelHelper, self).__init__(
            order='NCHW',  # this is only relevant for convolutional networks
            init_params=init_params,
        )
        self.non_trainable_params = []

    def AddParam(self, name, init=None, init_value=None, trainable=True):
        """Adds a parameter to the model's net and it's initializer if needed

        Args:
            init: a tuple (<initialization_op_name>, <initialization_op_kwargs>)
            init_value: int, float or str. Can be used instead of `init` as a
                simple constant initializer
            trainable: bool, whether to compute gradient for this param or not
        """
        if init_value is not None:
            assert init is None
            assert type(init_value) in [int, float, str]
            init = ('ConstantFill', dict(
                shape=[1],
                value=init_value,
            ))

        if self.init_params:
            param = self.param_init_net.__getattr__(init[0])(
                [],
                name,
                **init[1]
            )
        else:
            param = self.net.AddExternalInput(name)

        if trainable:
            self.params.append(param)
        else:
            self.non_trainable_params.append(param)

        return param


def rnn_unidirectional_encoder(
    model,
    embedded_inputs,
    input_lengths,
    initial_hidden_state,
    initial_cell_state,
    embedding_size,
    encoder_num_units,
    use_attention,
    num_layers=1,
):
    """ Unidirectional (forward pass) LSTM encoder."""

    input_blob=embedded_inputs
    dim_in=embedding_size
    dim_out=encoder_num_units

    final_hidden_states=[]
    final_cell_states=[]
    for l in range(num_layers):
        outputs, final_hidden_state, _, final_cell_state = recurrent.LSTM(
            model=model,
            input_blob=input_blob,
            seq_lengths=input_lengths,
            initial_states=(initial_hidden_state, initial_cell_state),
            dim_in=dim_in,
            dim_out=dim_out,
            scope='encoder/layer_{}'.format(l),
            outputs_with_grads=([0] if use_attention else [1, 3]),
        )
        final_hidden_states.append(final_hidden_state)
        final_cell_states.append(final_cell_state)
        input_blob=outputs
        dim_in=dim_out

    return outputs, final_hidden_states, final_cell_states


def rnn_bidirectional_encoder(
    model,
    embedded_inputs,
    input_lengths,
    initial_hidden_state,
    initial_cell_state,
    embedding_size,
    encoder_num_units,
    use_attention,
    num_layers=1,
):
    """ Bidirectional (forward pass and backward pass) Stacked-LSTM encoder."""

    input_blob=embedded_inputs
    dim_in=embedding_size
    dim_out=encoder_num_units

    for l in range(num_layers):
        # Forward pass
        (
            outputs_fw,
            final_hidden_state_fw,
            _,
            final_cell_state_fw,
        ) = recurrent.LSTM(
            model=model,
            input_blob=input_blob,
            seq_lengths=input_lengths,
            initial_states=(initial_hidden_state, initial_cell_state),
            dim_in=dim_in,
            dim_out=dim_out,
            scope='forward_encoder/layer_{}'.format(l),
            outputs_with_grads=([0] if use_attention else [1, 3]),
        )

        # Backward pass
        reversed_input_blob = model.net.ReversePackedSegs(
            [input_blob, input_lengths],
            ['reversed_input_blob'],
        )

        (
            outputs_bw,
            final_hidden_state_bw,
            _,
            final_cell_state_bw,
        ) = recurrent.LSTM(
            model=model,
            input_blob=reversed_input_blob,
            seq_lengths=input_lengths,
            initial_states=(initial_hidden_state, initial_cell_state),
            dim_in=dim_in,
            dim_out=dim_out,
            scope='backward_encoder/layer_{}'.format(l),
            outputs_with_grads=([0] if use_attention else [1, 3]),
        )

        outputs_bw = model.net.ReversePackedSegs(
            [outputs_bw, input_lengths],
            ['outputs_bw'],
        )

        # Concatenate forward and backward results
        outputs, _ = model.net.Concat(
            [outputs_fw, outputs_bw],
            ['outputs', 'outputs_dim'],
            axis=2,
        )

        final_hidden_state, _ = model.net.Concat(
            [final_hidden_state_fw, final_hidden_state_bw],
            ['final_hidden_state', 'final_hidden_state_dim'],
            axis=2,
        )

        final_cell_state, _ = model.net.Concat(
            [final_cell_state_fw, final_cell_state_bw],
            ['final_cell_state', 'final_cell_state_dim'],
            axis=2,
        )

        input_blob=outputs
        dim_in=dim_out

    return outputs, final_hidden_state, final_cell_state
