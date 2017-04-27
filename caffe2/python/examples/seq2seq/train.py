## @package seq2seq
# Module caffe2.python.examples.seq2seq
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import random
import sys
from timeit import default_timer as timer

from caffe2.python import workspace
import model, data

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stderr))

def run_seq2seq_model(args, model_params=None):
    (source_vocab, target_vocab,
     train_data, eval_data) = data.get_data(args)

    with model.Seq2SeqModelCaffe2(
        model_params=model_params,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        num_gpus=args.num_gpus,
        num_cpus=20,
        optimizer=args.optimizer,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        init_decoder=args.init_decoder,
    ) as model_obj:
        model_obj.initialize_from_scratch()

        with open("seq2seq.pbtxt", "w") as fid:
            fid.write(str(model_obj.model.net.Proto()))
        with open("seq2seq_init.pbtxt", "w") as fid:
            fid.write(str(model_obj.model.param_init_net.Proto()))
        with open("seq2seq_forward.pbtxt", "w") as fid:
            fid.write(str(model_obj.forward_net.Proto()))

        train_sequences_per_epoch = sum([len(train_data[key][0])
                                   for key in train_data])
        train_iterations_per_epoch = int(np.ceil(train_sequences_per_epoch / args.batch_size))
        train_encoder_tokens_per_epoch = sum([np.sum(train_data[key][1])
                                        for key in train_data])
        train_decoder_tokens_per_epoch = sum([np.sum(train_data[key][3])
                                        for key in train_data])
        train_total_tokens_with_padding_per_epoch = sum([
            train_data[key][0].size + train_data[key][2].size
            for key in train_data])

        display_interval = int(np.ceil(train_iterations_per_epoch / 100.0))
        test_interval = int(np.ceil(train_iterations_per_epoch / 10.0))

        eval_sequences_per_epoch = sum([len(eval_data[key][0])
                                   for key in eval_data])
        eval_iterations_per_epoch = int(np.ceil(eval_sequences_per_epoch / args.batch_size))
        eval_encoder_tokens_per_epoch = sum([np.sum(eval_data[key][1])
                                        for key in eval_data])
        eval_decoder_tokens_per_epoch = sum([np.sum(eval_data[key][3])
                                        for key in eval_data])
        eval_total_tokens_with_padding_per_epoch = sum([
            eval_data[key][0].size + eval_data[key][2].size
            for key in eval_data])

        print("Training stats:")
        print("    Iterations per epoch:     ", train_iterations_per_epoch)
        print("    Sequences per epoch:      ", train_sequences_per_epoch)
        print("    Encoder tokens per epoch: ", train_encoder_tokens_per_epoch)
        print("    Decoder tokens per epoch: ", train_decoder_tokens_per_epoch)
        print("    Total tokens per epoch:   ",
              train_encoder_tokens_per_epoch + train_decoder_tokens_per_epoch)
        print("    Tokens+padding per epoch: ", train_total_tokens_with_padding_per_epoch)
        print("    display_interval:         ", display_interval)
        print("    test_interval:            ", test_interval)
        print("    learning rate:            ", float(workspace.FetchBlob("learning_rate")))

        print("Evaluation stats:")
        print("    Iterations per epoch:     ", eval_iterations_per_epoch)
        print("    Sequences per epoch:      ", eval_sequences_per_epoch)
        print("    Encoder tokens per epoch: ", eval_encoder_tokens_per_epoch)
        print("    Decoder tokens per epoch: ", eval_decoder_tokens_per_epoch)
        print("    Total tokens per epoch:   ",
              eval_encoder_tokens_per_epoch + eval_decoder_tokens_per_epoch)
        print("    Tokens+padding per epoch: ", eval_total_tokens_with_padding_per_epoch)

        epoch_train_loss = np.zeros(args.epochs, dtype=np.float32)
        epoch_eval_loss = np.zeros(args.epochs, dtype=np.float32)
        epoch_train_perplexity = []
        epoch_eval_perplexity = []
        for epoch in range(args.epochs):
            # For display
            epoch_start = timer()
            current_iter = 0
            last_display_iter = 0
            display_time = 0
            display_loss = 0
            display_sequences = 0
            display_encoder_tokens = 0
            display_decoder_tokens = 0
            display_tokens_with_padding = 0

            for batch in data.iterate_epoch(
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
                epoch_train_loss[epoch] += loss
                display_sequences += len(batch[0])
                display_encoder_tokens += np.sum(batch[1])
                display_decoder_tokens += np.sum(batch[3])
                display_tokens_with_padding += batch[0].size + batch[2].size
                current_epoch = epoch + float(current_iter) / train_iterations_per_epoch

                # Display
                if current_iter % display_interval == 0:
                    perplexity = pow(2, display_loss / display_decoder_tokens)
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
                          ((display_encoder_tokens + display_decoder_tokens)
                           / display_time,))
                    print("    Tokens+padding/second: %f" %
                          (display_tokens_with_padding / display_time,))
                    last_display_iter = current_iter
                    display_time = 0
                    display_loss = 0
                    display_sequences = 0
                    display_tokens = 0
                    display_padding = 0
                    display_encoder_tokens = 0
                    display_decoder_tokens = 0
                    display_tokens_with_padding = 0

                if current_iter % test_interval == 0:
                    print("\nEvaluating model ...")
                    eval_start = timer()
                    eval_iterations = 0
                    eval_loss = 0
                    eval_sequences = 0
                    eval_encoder_tokens = 0
                    eval_decoder_tokens = 0
                    eval_tokens_with_padding = 0
                    for batch in data.iterate_epoch(
                            eval_data, args.batch_size):
                        loss = model_obj.step(batch=batch, forward_only=True)
                        eval_iterations += 1
                        eval_loss += loss
                        eval_sequences += len(batch[0])
                        eval_encoder_tokens += np.sum(batch[1])
                        eval_decoder_tokens += np.sum(batch[3])
                        eval_tokens_with_padding += batch[0].size + batch[2].size

                    eval_time = timer() - eval_start
                    epoch_eval_loss[epoch] = eval_loss

                    perplexity = pow(2, eval_loss / eval_decoder_tokens)
                    print("    Displaying after %d iterations, %f seconds" %
                          (eval_iterations, eval_time))
                    print("    Evaluation loss=%f, perplexity=%f" %
                          (eval_loss, perplexity))
                    print("    Iterations/second:     %f" %
                          (eval_iterations / eval_time,))
                    print("    Sequences/second:      %f" %
                          (eval_sequences / eval_time,))
                    print("    Tokens/second:         %f" %
                          ((eval_encoder_tokens + eval_decoder_tokens)
                           / eval_time,))
                    print("    Tokens+padding/second: %f" %
                          (eval_tokens_with_padding / eval_time,))
                    print()

            train_eval_epoch_time = timer() - epoch_start
            epoch_train_perplexity.append(
                pow(2, epoch_train_loss[epoch] / train_decoder_tokens_per_epoch))
            epoch_eval_perplexity.append(
                pow(2, epoch_eval_loss[epoch] / eval_decoder_tokens_per_epoch))
            print("\nEpoch %d finished in %d seconds.\n" %
                  (epoch + 1, int(round(train_eval_epoch_time))))
            print("    Training loss=%f, perplexity=%f" %
                  (epoch_train_loss[epoch], epoch_train_perplexity[epoch]))
            print("    Evaluation loss=%f, perplexity=%f" %
                  (epoch_eval_loss[epoch], epoch_eval_perplexity[epoch]))
            print()

            # Change LR if needed
            if (epoch >= args.start_decay_at) or\
                    (epoch > 0 and epoch_eval_perplexity[epoch] > epoch_eval_perplexity[epoch-1]):
                current_lr = float(workspace.FetchBlob("learning_rate"))
                adjusted_lr = current_lr * args.learning_rate_decay
                workspace.FeedBlob("learning_rate", np.array([adjusted_lr], dtype=np.float32))
                print("    Changing learning rate from {} to {}.".format(current_lr, adjusted_lr))

def train(args):
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
    data.addParserArguments(parser)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of iterations over training data')
    parser.add_argument('--learning-rate', type=float, default=1.0,
                        help='Learning rate')
    parser.add_argument('--learning-rate-decay', type=float, default=0.5,
                        help='Learning rate')
    parser.add_argument('--start-decay-at', type=int, default=8,
                        help='Learning rate')
    parser.add_argument('--max-gradient-norm', type=float, default=1.0,
                        help='Max global norm of gradients at the end of each '
                        'backward pass. We do clipping to match the number.')
    parser.add_argument('--use-bidirectional-encoder', action='store_true',
                        help='Set flag to use bidirectional recurrent network '
                        'in encoder')
    parser.add_argument('--use-attention', action='store_true',
                        help='Set flag to use seq2seq with attention model')
    parser.add_argument('--num-encoder-layers', type=int, default=1,
                        help='Number of RNN layers in encoder')
    parser.add_argument('--num-decoder-layers', type=int, default=1,
                        help='Number of RNN layers in decoder')
    parser.add_argument('--init-decoder', type=int, default=0,
                        help='If use_attention, initialize decoder states with '
                        ' the last encoder states. If 0, the initial decoder '
                        'states are set to zero.')
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

    train(args)


if __name__ == '__main__':
    main()
