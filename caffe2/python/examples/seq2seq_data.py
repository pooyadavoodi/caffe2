#!/usr/bin/env python
from __future__ import division, print_function

import argparse
from collections import Counter, defaultdict
import itertools
import os.path
import random
import sys
from timeit import default_timer as timer

import numpy as np
import six

import seq2seq_cache as cache


_PAD_ID = 0
_GO_ID = 1
_EOS_ID = 2
EOS = '<EOS>'
UNK = '<UNK>'
GO = '<GO>'
PAD = '<PAD>'


def tokenize_line(line):
    """Make a list of tokens from a line of input."""
    return line.strip().split()


def gen_vocab(filename, max_size):
    """Parse the file at FILENAME and return a vocabulary dictionary mapping
    tokens to numeric identifiers, with maximum vocabulary size of MAX_SIZE.
    """
    counter = Counter()
    with open(filename, 'r') as infile:
        for line in infile:
            for token in tokenize_line(line):
                counter[token] += 1

    vocab = {PAD: 0, GO: 1, EOS: 2, UNK: 3}
    i = 4
    max_size += i
    # sort by (count, key)
    for key, count in sorted(six.iteritems(counter), reverse=True,
                             key=lambda item: (item[1], item[0])):
        vocab[key] = i
        i += 1
        if i >= max_size:
            break
    assert len(vocab.keys()) <= max_size
    return vocab


def encode_line(line, vocab):
    """Convert LINE into a np.ndarray, where each element is the numeric
    representation of a token according to VOCAB.
    """
    tokens = tokenize_line(line)
    data = np.zeros(len(tokens), dtype=np.int32)
    for i, token in enumerate(tokens):
        data[i] = vocab.get(token, vocab[UNK])
    return data


def process_textfiles(
        source_filename, target_filename,
        source_vocab, target_vocab,
        source_max_len=None, target_max_len=None):
    """Processes a pair of (source, target) textfiles. Returns a dict which
    maps a (source_length, target_length) tuple of ints to a [source_inputs,
    source_lengths, target_inputs, target_lengths] list of ndarrays.
    """
    # keys: (source_len, target_len)
    # values: [(source_encoding, target_encoding), ...]
    data = defaultdict(list)
    with open(source_filename, 'r') as source_file:
        with open(target_filename, 'r') as target_file:
            for source_line, target_line in itertools.izip(
                    source_file, target_file):
                source_encoding = encode_line(source_line, source_vocab)
                target_encoding = encode_line(target_line, target_vocab)
                if (len(source_encoding) == 0 or len(target_encoding) == 0
                        or
                        (source_max_len is not None and
                         len(source_encoding) > source_max_len)
                        or
                        (target_max_len is not None and
                         len(target_encoding) > target_max_len)
                ):
                    continue  # Skip this line - an encoding is too long
                key = (len(source_encoding), len(target_encoding))
                data[key].append((source_encoding, target_encoding))
    # keys: (source_len, target_len)
    # values: [source_inputs, source_lengths, target_inputs, target_lengths]
    return dict([(
        key,
        [np.vstack(item[0] for item in value),
         np.full(len(value), key[0], dtype=np.int32),
         np.vstack(item[1] for item in value),
         np.full(len(value), key[1], dtype=np.int32),
         ]
    ) for key, value in six.iteritems(data)])


def get_sizes_matrix(data):
    """Return a 2D matrix where matrix[i,j] returns the number of samples
    with len(source)==i and len(target)==j.
    """
    max_x = 0
    max_y = 1
    for key, value in six.iteritems(data):
        max_x = max(max_x, key[0])
        max_y = max(max_y, key[1])
    matrix = np.zeros((max_x + 1, max_y + 1), dtype=np.int32)
    for key, value in six.iteritems(data):
        matrix[key] = len(value[0])
    return matrix


def get_token_count(matrix):
    """Returns the total number of tokens as defined by a sizes matrix."""
    tokens = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            tokens += i * j * matrix[i, j]
    return tokens


def auto_trim_data(data, fraction):
    """Returns a trimmed-down version of data. Roughly FRACTION of the data
    has been removed by trimming away the longest sequences.
    """
    matrix = get_sizes_matrix(data)
    cutoff = np.sum(matrix) * fraction
    i, j = matrix.shape

    # Decrease I until CUTOFF is reached
    stripped = 0
    while i > 1:
        s = np.sum(matrix[i - 1,:])
        if stripped + s > cutoff:
            break
        i -= 1
        stripped += s

    # Decrease J until CUTOFF is reached
    stripped = 0
    while j > 1:
        s = np.sum(matrix[:, j - 1])
        if stripped + s > cutoff:
            break
        j -= 1
        stripped += s

    return {key: value for key, value in six.iteritems(data)
            if key[0] <= i and key[1] <= j}


def pad_tensor(tensor, new_size):
    """Add padding to tensor, to bring sequence length up to NEW_SIZE."""
    if tensor.shape[1] == new_size:
        return tensor
    return np.hstack([
        tensor,
        np.full((tensor.shape[0], new_size - tensor.shape[1]), _PAD_ID,
                dtype=np.int32)
    ])


def auto_bucketize_data(data, batch_size, epochs=1):
    """Automatically bucketize DATA, returning the same data, but with
    small buckets merged into buckets with larger dimensions, such that
    each bucket contains a reasonable minimum amount of data.
    """
    matrix = get_sizes_matrix(data)
    sample_count = np.sum(matrix)
    bucket_size = min(
        # No larger than sample_count
        sample_count,
        max(
            # Large enough to make batch remainders a non-issue
            batch_size ** 2,
            # Large enough for good shuffling across batches
            batch_size * epochs,
        )
    )

    def _merge_buckets(old_shape, new_shape):
        """Utitity function - merge two buckets."""
        assert old_shape != new_shape
        (source_inputs, source_lengths,
         target_inputs, target_lengths) = data.pop(old_shape)
        source_inputs = pad_tensor(source_inputs, new_shape[0])
        target_inputs = pad_tensor(target_inputs, new_shape[1])
        if new_shape not in data:
            data[new_shape] = [source_inputs, source_lengths,
                               target_inputs, target_lengths]
        else:
            data[new_shape] = [
                np.vstack([data[new_shape][0], source_inputs]),
                np.hstack([data[new_shape][1], source_lengths]),
                np.vstack([data[new_shape][2], target_inputs]),
                np.hstack([data[new_shape][3], target_lengths])]
        matrix[new_shape] += matrix[old_shape]
        matrix[old_shape] = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            sz = matrix[i, j]
            if sz == 0:
                continue  # Bucket is empty
            if sz >= bucket_size:
                continue  # Bucket is already full enough

            i_is_max = i + 1 >= matrix.shape[0]
            j_is_max = j + 1 >= matrix.shape[1]
            if i_is_max and j_is_max:
                # This is the sloppiest part of the algorithm. Merge nearby
                # buckets into the last bucket until it's full.
                for sum_of_indices in range(i + j, -1, -1):
                    for i2 in range(sum_of_indices + 1):
                        j2 = sum_of_indices - i2
                        if ((i2, j2) != (i, j) and i2 < matrix.shape[0] and
                                j2 < matrix.shape[1] and matrix[i2, j2] > 0):
                            _merge_buckets((i2, j2), (i, j))
                            if matrix[i, j] >= bucket_size:
                                break
                    if matrix[i, j] >= bucket_size:
                        break
            elif i_is_max:
                _merge_buckets((i, j), (i, j+1))
            elif j_is_max:
                _merge_buckets((i, j), (i+1, j))
            else:
                # Add to whichever bucket is larger: (i+1, j) or (i, j+1)
                sz_a = matrix[i + 1, j]
                sz_b = matrix[i, j + 1]
                if sz_a > sz_b or (sz_a == sz_b and i <= j):
                    _merge_buckets((i, j), (i+1, j))
                else:
                    _merge_buckets((i, j), (i, j+1))
    return data


class MyTimer():
    """Utility class for timing potentially slow code paths."""
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        sys.stdout.write(self.msg + ' ... ')
        sys.stdout.flush()
        self.start = timer()

    def __exit__(self, *args):
        print('DONE in %fs' % (timer() - self.start,))
        return False

def get_data(args):
    """Returns (train_data, test_data)."""
    # Argument cleanup
    train_source = os.path.realpath(args.source_corpus)
    train_target = os.path.realpath(args.target_corpus)
    test_source = os.path.realpath(args.source_corpus_eval)
    test_target = os.path.realpath(args.target_corpus_eval)
    try:
        batch_size = args.batch_size
    except AttributeError:
        batch_size = 32
    try:
        epochs = args.epochs
    except AttributeError:
        epochs = 10

    # Get source vocab
    source_vocab_cache_id = cache.get_id(train_source, args.max_vocab_size)
    if cache.exists(source_vocab_cache_id):
        with MyTimer('cache.load(source_vocab)'):
            source_vocab = cache.load(source_vocab_cache_id)
    else:
        with MyTimer('gen_vocab(source)'):
            source_vocab = gen_vocab(train_source, args.max_vocab_size)
        cache.store(source_vocab_cache_id, source_vocab)
    print('    Vocab size:', len(source_vocab))

    # Get target vocab
    target_vocab_cache_id = cache.get_id(train_target, args.max_vocab_size)
    if cache.exists(target_vocab_cache_id):
        with MyTimer('cache.load(target_vocab)'):
            target_vocab = cache.load(target_vocab_cache_id)
    else:
        with MyTimer('gen_vocab(target)'):
            target_vocab = gen_vocab(train_target, args.max_vocab_size)
        cache.store(target_vocab_cache_id, target_vocab)
    print('    Vocab size:', len(target_vocab))

    # Get training data
    train_data_cache_id = cache.get_id(
        train_source, train_target, args.max_vocab_size,
        args.max_sequence_length, args.trim_fraction)
    if cache.exists(train_data_cache_id):
        with MyTimer('cache.load(train_data)'):
            train_data = cache.load(train_data_cache_id)
    else:
        with MyTimer('process_textfiles(train)'):
            train_data = process_textfiles(
                train_source, train_target,
                source_vocab, target_vocab,
                source_max_len=args.max_sequence_length,
                target_max_len=args.max_sequence_length,
            )
        if args.max_sequence_length is None:
            with MyTimer('auto_trim_data(train_data)'):
                train_data = auto_trim_data(train_data, args.trim_fraction)
        cache.store(train_data_cache_id, train_data)
    matrix = get_sizes_matrix(train_data)
    print('    Training samples:', np.sum(matrix))
    print('    Max source length:', matrix.shape[0])
    print('    Max target length:', matrix.shape[1])
    print('    Token count:', get_token_count(matrix))
    print('    Bucket count:', np.count_nonzero(matrix))

    with MyTimer('auto_bucketize_data(train_data)'):
        train_data = auto_bucketize_data(train_data, batch_size, epochs)
    matrix = get_sizes_matrix(train_data)
    print('    Token count:', get_token_count(matrix))
    print('    Bucket count:', np.count_nonzero(matrix))

    # Get test data
    test_data_cache_id = cache.get_id(
        test_source, test_target, args.max_vocab_size,
        args.max_sequence_length, args.trim_fraction)
    if cache.exists(test_data_cache_id):
        with MyTimer('cache.load(test_data)'):
            test_data = cache.load(test_data_cache_id)
    else:
        with MyTimer('process_textfiles(test)'):
            test_data = process_textfiles(
                test_source, test_target,
                source_vocab, target_vocab,
                source_max_len=args.max_sequence_length,
                target_max_len=args.max_sequence_length,
            )
        if args.max_sequence_length is None:
            with MyTimer('auto_trim_data(test_data)'):
                test_data = auto_trim_data(test_data, args.trim_fraction)
        cache.store(test_data_cache_id, test_data)
    matrix = get_sizes_matrix(test_data)
    print('    Training samples:', np.sum(matrix))
    print('    Max source length:', matrix.shape[0])
    print('    Max target length:', matrix.shape[1])
    print('    Token count:', get_token_count(matrix))
    print('    Bucket count:', np.count_nonzero(matrix))

    with MyTimer('auto_bucketize_data(test_data)'):
        test_data = auto_bucketize_data(test_data, batch_size)
    matrix = get_sizes_matrix(test_data)
    print('    Token count:', get_token_count(matrix))
    print('    Bucket count:', np.count_nonzero(matrix))

    return (source_vocab, target_vocab,
            train_data, test_data)


def iterate_epoch(data, batch_size, shuffle=False):
    """Iterate through DATA once, yielding batches."""
    keys = data.keys()
    if shuffle:
        random.shuffle(keys)
    else:
        keys = sorted(keys)

    batches = []
    for key in keys:
        count = len(data[key][0])
        indices = range(count)
        if shuffle:
            random.shuffle(indices)
        for i in range(count // batch_size):
            batches.append((key, indices[i * batch_size:(i + 1) * batch_size]))
        if count % batch_size != 0:
            # Last batch wraps around
            batches.append((key,
                            indices[(count // batch_size) * batch_size:] +
                            indices[:-count % batch_size]))
    if shuffle:
        random.shuffle(batches)
    for key, indices in batches:
        yield (data[key][0][indices], data[key][1][indices],
               data[key][2][indices], data[key][3][indices])


def addParserArguments(parser):
    """Add arguments to an argparse.ArgumentParser."""
    parser.add_argument('--source-corpus', required=True,
                        help='Path to a textfile containing the training '
                        'source corpus. Each line in the file is a sequence.')
    parser.add_argument('--target-corpus', required=True,
                        help='Path to a textfile containing the training '
                        'target corpus. Each line in the file is a sequence.')
    parser.add_argument('--source-corpus-eval', required=True,
                        help='Path to a textfile containing the evaluation '
                        'source corpus. Each line in the file is a sequence.')
    parser.add_argument('--target-corpus-eval', required=True,
                        help='Path to a textfile containing the evaluation '
                        'target corpus. Each line in the file is a sequence.')
    parser.add_argument('--max-vocab-size', type=int, default=50000,
                        help='Maxiumum vocabulary size.')
    parser.add_argument('--max-sequence-length', type=int,
                        help='Maxiumum length of any sequence.')
    parser.add_argument('--trim-fraction', type=float, default=0.05,
                        help='Trim down the corpus size by removing examples '
                        'with the largest number of tokens. Not used if '
                        '--max-sequence-length is set.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Caffe2: Seq2Seq data utilities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    addParserArguments(parser)
    args = parser.parse_args()

    # Example of how to get the data
    (_, _, train_data, test_data) = get_data(args)

    # Example of how to iterate over the data
    with MyTimer('next(iterate_epoch(train_data))'):
        train_gen = iterate_epoch(train_data, 32, shuffle=True)
        train_batch1 = next(train_gen)
    print('    First train batch:', [x.shape for x in train_batch1])
    with MyTimer('next(iterate_epoch(test_data))'):
        test_gen = iterate_epoch(test_data, 32)
        test_batch1 = next(test_gen)
    print('    First test batch:', [x.shape for x in test_batch1])
