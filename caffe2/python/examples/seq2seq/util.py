## @package seq2seq_util
# Module caffe2.python.examples.seq2seq_util
""" A bunch of util functions to build Seq2Seq models with Caffe2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import caffe2.proto.caffe2_pb2 as caffe2_pb2
from caffe2.python.cnn import CNNModelHelper
from caffe2.python import core, workspace, recurrent

def rnn_unidirectional_encoder(
    model,
    embedded_inputs,
    input_lengths,
    initial_hidden_state,
    initial_cell_state,
    embedding_size,
    encoder_num_units,
    use_attention,
    num_layers,
    dropout,
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
        if(dropout != None) and (l != num_layers-1):
          outputs = model.Dropout(outputs, str(outputs), ratio=dropout)
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
    num_layers,
    dropout,
):
    """ Bidirectional (forward pass and backward pass) Stacked-LSTM encoder."""

    input_blob=embedded_inputs
    dim_in=embedding_size
    dim_out=encoder_num_units

    final_hidden_states=[]
    final_cell_states=[]
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
        if(dropout != None) and (l != num_layers-1):
          outputs = model.Dropout(outputs, str(outputs), ratio=dropout)
        final_hidden_states.append(final_hidden_state)
        final_cell_states.append(final_cell_state)
        input_blob=outputs
        dim_in=dim_out

    return outputs, final_hidden_states, final_cell_states




def build_embedding_encoder(
    model,
    encoder_params,
    inputs,
    input_lengths,
    vocab_size,
    embeddings,
    embedding_size,
    use_attention,
    num_layers,
    dropout,
    num_gpus,
    scope=None,
):
    with core.NameScope(scope or ''):
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

    assert len(encoder_params['encoder_layer_configs']) == 1
    encoder_num_units = (
        encoder_params['encoder_layer_configs'][0]['num_units']
    )
    with core.NameScope(scope or ''):
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
        if encoder_params['use_bidirectional_encoder']:
            rnn_encoder_func = rnn_bidirectional_encoder
            encoder_output_dim = 2 * encoder_num_units
        else:
            rnn_encoder_func = rnn_unidirectional_encoder
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
        num_layers,
        dropout,
    )
    weighted_encoder_outputs = None

    return (
        encoder_outputs,
        weighted_encoder_outputs,
        final_encoder_hidden_state,
        final_encoder_cell_state,
        encoder_output_dim,
    )

def build_initial_rnn_decoder_states(
    model,
    encoder_num_units,
    decoder_num_units,
    final_encoder_hidden_states,
    final_encoder_cell_states,
    num_layers,
    init_decoder,
    use_attention,
):
    # Initializing RNN states
    if (use_attention == False) and (init_decoder == 1):
        raise ValueError("Use init_decoder=1 only without attention.")
    if (init_decoder == 1) and (encoder_num_units != decoder_num_units):
        raise ValueError("Use init_decoder=1 only if num-units of encoder and decoder are equal.")
    if(init_decoder):
      assert len(final_encoder_hidden_states) == num_layers
      assert len(final_encoder_cell_states) == num_layers

    decoder_initial_hidden_states=[]
    decoder_initial_cell_states=[]
    for l in range(num_layers):
        if use_attention:
            if init_decoder == 1:
                decoder_initial_hidden_state = final_encoder_hidden_states[l]
                decoder_initial_cell_state = final_encoder_cell_states[l]
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
        else:
            decoder_initial_hidden_state = model.FC(
                final_encoder_hidden_states[l],
                'decoder_initial_hidden_state',
                encoder_num_units,
                decoder_num_units,
                axis=2,
            )
            decoder_initial_cell_state = model.FC(
                final_encoder_cell_states[l],
                'decoder_initial_cell_state',
                encoder_num_units,
                decoder_num_units,
                axis=2,
            )
        decoder_initial_hidden_states.append(decoder_initial_hidden_state)
        decoder_initial_cell_states.append(decoder_initial_hidden_state)

    if use_attention:
        initial_attention_weighted_encoder_context = (
            model.param_init_net.ConstantFill(
                [],
                'initial_attention_weighted_encoder_context',
                shape=[encoder_num_units],
                value=0.0,
            )
        )
        return (decoder_initial_hidden_states,
                decoder_initial_cell_states,
                initial_attention_weighted_encoder_context)
    else:
        return (decoder_initial_hidden_states,
                decoder_initial_cell_states)

def output_projection(
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
        shape=[target_vocab_size, decoder_output_size],
    )

    output_projection_b = model.param_init_net.XavierFill(
        [],
        'output_projection_b',
        shape=[target_vocab_size],
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


