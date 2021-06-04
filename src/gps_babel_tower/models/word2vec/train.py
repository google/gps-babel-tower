"""Train a word2vec model to obtain word embedding vectors.

There are a total of four combination of architectures and training algorithms
that the model can be trained with:

architecture:
  - skip_gram
  - cbow (continuous bag-of-words)

training algorithm
  - negative_sampling
  - hierarchical_softmax
"""
import os

import tensorflow as tf
import numpy as np
import argparse

from .dataset import WordTokenizer
from .dataset import Word2VecDatasetBuilder
from .model import Word2VecModel
from .word_vectors import WordVectors

from . import utils



def main(args):  
  arch = args.arch
  algm = args.algm
  epochs = args.epochs
  batch_size = args.batch_size
  max_vocab_size = args.max_vocab_size
  min_count = args.min_count
  sample = args.sample
  window_size = args.window_size
  hidden_size = args.hidden_size
  negatives = args.negatives
  power = args.power
  alpha = args.alpha
  min_alpha = args.min_alpha
  add_bias = args.add_bias
  log_per_steps = args.log_per_steps
  input_path = args.input_path
  job_dir = args.job_dir
  log_dir = f'{job_dir}/logs'
  
  tf.io.gfile.makedirs(job_dir)
  
  tokenizer = WordTokenizer(
      max_vocab_size=max_vocab_size, min_count=min_count, sample=sample)
  tokenizer.build_vocab(input_path)

  builder = Word2VecDatasetBuilder(tokenizer,
                                   arch=arch,
                                   algm=algm,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   window_size=window_size)
  dataset = builder.build_dataset(input_path)
  word2vec = Word2VecModel(tokenizer.unigram_counts,
               arch=arch,
               algm=algm,
               hidden_size=hidden_size,
               batch_size=batch_size,
               negatives=negatives,
               power=power,
               alpha=alpha,
               min_alpha=min_alpha,
               add_bias=add_bias)

  train_step_signature = utils.get_train_step_signature(
      arch, algm, batch_size, window_size, builder._max_depth)
  optimizer = tf.keras.optimizers.SGD(1.0)
  
  log_writer = tf.summary.create_file_writer(log_dir)

  @tf.function(input_signature=train_step_signature)
  def train_step(inputs, labels, progress):
    
    with tf.GradientTape() as tape:
      loss = word2vec(inputs, labels)
    
    gradients = tape.gradient(loss, word2vec.trainable_variables)
  
    learning_rate = tf.maximum(alpha * (1 - progress[0]) +
        min_alpha * progress[0], min_alpha)

    if hasattr(gradients[0], '_values'):
      gradients[0]._values *= learning_rate
    else:
      gradients[0] *= learning_rate

    if hasattr(gradients[1], '_values'):
      gradients[1]._values *= learning_rate
    else:
      gradients[1] *= learning_rate

    if hasattr(gradients[2], '_values'):
      gradients[2]._values *= learning_rate
    else:
      gradients[2] *= learning_rate

    optimizer.apply_gradients(
        zip(gradients, word2vec.trainable_variables))

    return loss, learning_rate

  average_loss = 0.
  for step, (inputs, labels, progress) in enumerate(dataset):
    loss, learning_rate = train_step(inputs, labels, progress)
    
    average_loss += loss.numpy().mean()
    if step % log_per_steps == 0:
      if step > 0:
        average_loss /= log_per_steps
      print('step:', step, 'average_loss:', average_loss,
            'learning_rate:', learning_rate.numpy())
      average_loss = 0.
      with log_writer.as_default():
        tf.summary.scalar('loss', average_loss, step=step)

  syn0_final = word2vec.weights[0].numpy()
  np.save(os.path.join(args.job_dir, 'syn0_final'), syn0_final)
  with tf.io.gfile.GFile(os.path.join(args.job_dir, 'vocab.txt'), 'w') as f:
    for w in tokenizer.table_words:
      f.write(w + '\n')
  print('Word embeddings saved to', 
      os.path.join(args.job_dir, 'syn0_final.npy'))
  print('Vocabulary saved to', os.path.join(args.job_dir, 'vocab.txt'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--arch', default='skip_gram', help='Architecture (skip_gram or cbow).')
  parser.add_argument('--algm', default='negative_sampling', help='Training algorithm (negative_sampling or hierarchical_softmax).')
  parser.add_argument('--epochs', default=1, type=int, help='Num of epochs to iterate thru corpus.')
  parser.add_argument('--batch-size', default=256, type=int, help='Batch size.')
  parser.add_argument('--max-vocab-size', default=0, type=int, help='Maximum vocabulary size. If > 0, the top `max_vocab_size` most frequent words will be kept in vocabulary.')
  parser.add_argument('--min-count', default=10, type=int, help='Words whose counts < `min_count` will not be included in the vocabulary.')
  parser.add_argument('--sample', default=1e-3, type=float, help='Subsampling rate.')
  parser.add_argument('--window-size', default=10, type=int, help='Num of words on the left or right side of target word within a window.')
  parser.add_argument('--hidden-size', default=300, type=int, help='Length of word vector.')
  parser.add_argument('--negatives', default=5, type=int, help='Num of negative words to sample.')
  parser.add_argument('--power', default=0.75, type=float, help='Distortion for negative sampling.')
  parser.add_argument('--alpha', default=0.025, type=float, help='Initial learning rate.')
  parser.add_argument('--min-alpha', default=0.0001, type=float, help='Final learning rate.')
  parser.add_argument('--add-bias', default=True, type=bool, help='Whether to add bias term to dotproduct between syn0 and syn1 vectors.')
  parser.add_argument('--log-per-steps', default=10000, type=int, help='Every `log_per_steps` steps to log the value of loss to be minimized.')
  parser.add_argument('--input-path', required=True, nargs='+', help='Input paths for text files.')
  parser.add_argument('--job-dir', default='/tmp/word2vec', help='Job directory for checkpoints and export.')
  
  args, _ = parser.parse_known_args()
  main(args)
