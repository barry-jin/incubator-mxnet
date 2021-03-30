# from d2l import mxnet as d2l
# import math
# from mxnet import gluon, np
# import os
# import random

# #@save
# d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
#                        '319d85e578af0cdc590547f26231e4e31cdf1e42')

# #@save
# def read_ptb():
#     data_dir = d2l.download_extract('ptb')
#     with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
#         raw_text = f.read()
#     return [line.split() for line in raw_text.split('\n')]

# sentences = read_ptb()

# vocab = d2l.Vocab(sentences, min_freq=10)

# #@save
# def subsampling(sentences, vocab):
#     # Map low frequency words into <unk>
#     sentences = [[vocab.idx_to_token[vocab[tk]] for tk in line]
#                  for line in sentences]
#     # Count the frequency for each word
#     counter = d2l.count_corpus(sentences)
#     num_tokens = sum(counter.values())

#     # Return True if to keep this token during subsampling
#     def keep(token):
#         return(random.uniform(0, 1) <
#                math.sqrt(1e-4 / counter[token] * num_tokens))

#     # Now do the subsampling
#     return [[tk for tk in line if keep(tk)] for line in sentences]

# subsampled = subsampling(sentences, vocab)

# def compare_counts(token):
#     return (f'# of "{token}": '
#             f'before={sum([line.count(token) for line in sentences])}, '
#             f'after={sum([line.count(token) for line in subsampled])}')

# corpus = [vocab[line] for line in subsampled]

# #@save
# def get_centers_and_contexts(corpus, max_window_size):
#     centers, contexts = [], []
#     for line in corpus:
#         # Each sentence needs at least 2 words to form a "central target word
#         # - context word" pair
#         if len(line) < 2:
#             continue
#         centers += line
#         for i in range(len(line)):  # Context window centered at i
#             window_size = random.randint(1, max_window_size)
#             indices = list(range(max(0, i - window_size),
#                                  min(len(line), i + 1 + window_size)))
#             # Exclude the central target word from the context words
#             indices.remove(i)
#             contexts.append([line[idx] for idx in indices])
#     return centers, contexts

# tiny_dataset = [list(range(7)), list(range(7, 10))]

# all_centers, all_contexts = get_centers_and_contexts(corpus, 5)

# #@save
# class RandomGenerator:
#     """Draw a random int in [0, n] according to n sampling weights."""
#     def __init__(self, sampling_weights):
#         self.population = list(range(len(sampling_weights)))
#         self.sampling_weights = sampling_weights
#         self.candidates = []
#         self.i = 0

#     def draw(self):
#         if self.i == len(self.candidates):
#             self.candidates = random.choices(
#                 self.population, self.sampling_weights, k=10000)
#             self.i = 0
#         self.i += 1
#         return self.candidates[self.i-1]

# generator = RandomGenerator([2, 3, 4])

# #@save
# def get_negatives(all_contexts, corpus, K):
#     counter = d2l.count_corpus(corpus)
#     sampling_weights = [counter[i]**0.75 for i in range(len(counter))]
#     all_negatives, generator = [], RandomGenerator(sampling_weights)
#     for contexts in all_contexts:
#         negatives = []
#         while len(negatives) < len(contexts) * K:
#             neg = generator.draw()
#             # Noise words cannot be context words
#             if neg not in contexts:
#                 negatives.append(neg)
#         all_negatives.append(negatives)
#     return all_negatives

# all_negatives = get_negatives(all_contexts, corpus, 5)

# #@save
# def batchify(data):
#     max_len = max(len(c) + len(n) for _, c, n in data)
#     centers, contexts_negatives, masks, labels = [], [], [], []
#     for center, context, negative in data:
#         cur_len = len(context) + len(negative)
#         centers += [center]
#         contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
#         masks += [[1] * cur_len + [0] * (max_len - cur_len)]
#         labels += [[1] * len(context) + [0] * (max_len - len(context))]
#     return (np.array(centers).reshape((-1, 1)), np.array(contexts_negatives),
#             np.array(masks), np.array(labels))

# x_1 = (1, [2, 2], [3, 3, 3, 3])
# x_2 = (1, [2, 2, 2], [3, 3])
# batch = batchify((x_1, x_2))

# names = ['centers', 'contexts_negatives', 'masks', 'labels']


# def load_data_ptb(batch_size, max_window_size, num_noise_words):
#     num_workers = d2l.get_dataloader_workers()
#     sentences = read_ptb()
#     vocab = d2l.Vocab(sentences, min_freq=10)
#     subsampled = subsampling(sentences, vocab)
#     corpus = [vocab[line] for line in subsampled]
#     all_centers, all_contexts = get_centers_and_contexts(
#         corpus, max_window_size)
#     all_negatives = get_negatives(all_contexts, corpus, num_noise_words)
#     dataset = gluon.data.ArrayDataset(
#         all_centers, all_contexts, all_negatives)
#     data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
#                                       batchify_fn=batchify,
#                                       num_workers=num_workers)
#     return data_iter, vocab

# data_iter, vocab = load_data_ptb(512, 5, 5)
# for batch in data_iter:
#     for name, data in zip(names, batch):
#         print(name, 'shape:', data.shape)
#     break


from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()

state = rnn_layer.begin_state(batch_size=batch_size)

X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)
