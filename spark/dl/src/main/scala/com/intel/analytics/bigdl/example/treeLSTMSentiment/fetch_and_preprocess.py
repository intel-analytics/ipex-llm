"""
Downloads the following:
- Glove vectors
- Stanford Sentiment Treebank (sentiment classification task)

Pre-process them to the easier using format

The script is modified from the pre-processing in original treelstm code,
which can be found at https://github.com/stanfordnlp/treelstm.
"""

from __future__ import print_function
import urllib2
import sys
import os
import shutil
import zipfile
import gzip
import glob	

#
# Trees and tree loading
#
class ConstTree(object):
    def __init__(self):
        self.left = None
        self.right = None

    def size(self):
        self.size = 1
        if self.left is not None:
            self.size += self.left.size()
        if self.right is not None:
            self.size += self.right.size()
        return self.size

    def set_spans(self):
        if self.word is not None:
            self.span = self.word
            return self.span

        self.span = self.left.set_spans()
        if self.right is not None:
            self.span += ' ' + self.right.set_spans()
        return self.span

    def get_labels(self, spans, labels, dictionary):
        if self.span in dictionary:
            spans[self.idx] = self.span
            labels[self.idx] = dictionary[self.span]
        if self.left is not None:
            self.left.get_labels(spans, labels, dictionary)
        if self.right is not None:
            self.right.get_labels(spans, labels, dictionary)

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    try:
        u = urllib2.urlopen(url)
    except:
        print("URL %s failed to open" %url)
        raise Exception
    try:
        f = open(filepath, 'wb')
    except:
        print("Cannot write %s" %filepath)
        raise Exception
    try:
        filesize = int(u.info().getheaders("Content-Length")[0])
    except:
        print("URL %s failed to report length" %url)
        raise Exception
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove vectors - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))

def download_sst(dirpath):
    if os.path.exists(dirpath):
        print('Found SST dataset - skip')
        return
    url = 'http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'
    parent_dir = os.path.dirname(dirpath)
    unzip(download(url, parent_dir))
    os.rename(
        os.path.join(parent_dir, 'stanfordSentimentTreebank'),
        os.path.join(parent_dir, 'sst'))
    shutil.rmtree(os.path.join(parent_dir, '__MACOSX')) # remove extraneous dir

def split(sst_dir, train_dir, dev_dir, test_dir):
    sents = load_sents(sst_dir)
    splits = load_splits(sst_dir)
    parents = load_parents(sst_dir)

    with open(os.path.join(train_dir, 'sents.txt'), 'w') as train, \
         open(os.path.join(dev_dir, 'sents.txt'), 'w') as dev, \
         open(os.path.join(test_dir, 'sents.txt'), 'w') as test, \
         open(os.path.join(train_dir, 'parents.txt'), 'w') as trainparents, \
         open(os.path.join(dev_dir, 'parents.txt'), 'w') as devparents, \
         open(os.path.join(test_dir, 'parents.txt'), 'w') as testparents:

        for sent, split, p in zip(sents, splits, parents):
            if split == 1:
                train.write(sent)
                train.write('\n')
                trainparents.write(p)
                trainparents.write('\n')
            elif split == 2:
                test.write(sent)
                test.write('\n')

                testparents.write(p)
                testparents.write('\n')
            else:
                dev.write(sent)
                dev.write('\n')
                devparents.write(p)
                devparents.write('\n')

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def load_sents(dirpath):
    sents = []
    with open(os.path.join(dirpath, 'SOStr.txt')) as sentsfile:
        for line in sentsfile:
            sent = ' '.join(line.split('|'))
            sents.append(sent.strip())
    return sents

def write_labels(dirpath, dictionary):
    print('Writing labels for trees in ' + dirpath)
    with open(os.path.join(dirpath, 'labels.txt'), 'w') as labels:
        # load constituency trees
        const_trees, toks = load_trees(dirpath)

        # write span labels
        for i in xrange(len(const_trees)):
            const_trees[i].set_spans()

            # const tree labels
            s, l = [], []
            for j in xrange(const_trees[i].size()):
                s.append(None)
                l.append(None)
            const_trees[i].get_labels(s, l, dictionary)
            labels.write(' '.join(map(str, l)) + '\n')

def load_dictionary(dirpath):
    labels = []
    with open(os.path.join(dirpath, 'sentiment_labels.txt')) as labelsfile:
        labelsfile.readline()
        for line in labelsfile:
            idx, rating = line.split('|')
            idx = int(idx)
            rating = float(rating)
            if rating <= 0.2:
                label = -2
            elif rating <= 0.4:
                label = -1
            elif rating > 0.8:
                label = +2
            elif rating > 0.6:
                label = +1
            else:
                label = 0
            labels.append(label)

    d = {}
    with open(os.path.join(dirpath, 'dictionary.txt')) as dictionary:
        for line in dictionary:
            s, idx = line.split('|')
            d[s] = labels[int(idx)]
    return d

def load_parents(dirpath):
    parents = []
    with open(os.path.join(dirpath, 'STree.txt')) as parentsfile:
        for line in parentsfile:
            p = ' '.join(line.split('|'))
            parents.append(p.strip())
    return parents

def load_splits(dirpath):
    splits = []
    with open(os.path.join(dirpath, 'datasetSplit.txt')) as splitfile:
        splitfile.readline()
        for line in splitfile:
            idx, split = line.split(',')
            splits.append(int(split))
    return splits

def load_trees(dirpath):
    const_trees, toks = [], []
    with open(os.path.join(dirpath, 'parents.txt')) as parentsfile, \
         open(os.path.join(dirpath, 'sents.txt')) as toksfile:
        parents, dparents = [], []
        for line in parentsfile:
            parents.append(map(int, line.split()))
        for line in toksfile:
            toks.append(line.strip().split())
        for i in xrange(len(toks)):
            const_trees.append(load_constituency_tree(parents[i], toks[i]))
    return const_trees, toks

def load_constituency_tree(parents, words):
    trees = []
    root = None
    size = len(parents)
    for i in xrange(size):
        trees.append(None)

    word_idx = 0
    for i in xrange(size):
        if not trees[i]:
            idx = i
            prev = None
            prev_idx = None
            word = words[word_idx]
            word_idx += 1
            while True:
                tree = ConstTree()
                parent = parents[idx] - 1
                tree.word, tree.parent, tree.idx = word, parent, idx
                word = None
                if prev is not None:
                    if tree.left is None:
                        tree.left = prev
                    else:
                        tree.right = prev
                trees[idx] = tree
                if parent >= 0 and trees[parent] is not None:
                    if trees[parent].left is None:
                        trees[parent].left = tree
                    else:
                        trees[parent].right = tree
                    break
                elif parent == -1:
                    root = tree
                    break
                else:
                    prev = tree
                    prev_idx = idx
                    idx = parent
    return root

def load_dependency_tree(parents):
    trees = []
    root = None
    size = len(parents)
    for i in xrange(size):
        trees.append(None)

    for i in xrange(size):
        if not trees[i]:
            idx = i
            prev = None
            prev_idx = None
            while True:
                tree = DepTree()
                parent = parents[idx] - 1

                # node is not in tree
                if parent == -2:
                    break

                tree.parent, tree.idx = parent, idx
                if prev is not None:
                    tree.children.append(prev)
                trees[idx] = tree
                if parent >= 0 and trees[parent] is not None:
                    trees[parent].children.append(tree)
                    break
                elif parent == -1:
                    root = tree
                    break
                else:
                    prev = tree
                    prev_idx = idx
                    idx = parent
    return root

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def load_dictionary(dirpath):
    labels = []
    with open(os.path.join(dirpath, 'sentiment_labels.txt')) as labelsfile:
        labelsfile.readline()
        for line in labelsfile:
            idx, rating = line.split('|')
            idx = int(idx)
            rating = float(rating)
            if rating <= 0.2:
                label = -2
            elif rating <= 0.4:
                label = -1
            elif rating > 0.8:
                label = +2
            elif rating > 0.6:
                label = +1
            else:
                label = 0
            labels.append(label)

    d = {}
    with open(os.path.join(dirpath, 'dictionary.txt')) as dictionary:
        for line in dictionary:
            s, idx = line.split('|')
            d[s] = labels[int(idx)]
    return d

if __name__ == '__main__':
    base_dir = '/tmp/.bigdl/'

   # data
    data_dir = os.path.join(base_dir, 'dataset')
    wordvec_dir = os.path.join(data_dir, 'glove')
    sst_dir = os.path.join(data_dir, 'sst')
    make_dirs([base_dir, data_dir])

    download_wordvecs(wordvec_dir)
    download_sst(sst_dir)

    train_dir = os.path.join(sst_dir, 'train')
    dev_dir = os.path.join(sst_dir, 'dev')
    test_dir = os.path.join(sst_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    # libraries
    lib_dir = os.path.join(base_dir, 'lib')


    print('=' * 80)
    print('Preprocessing Stanford Sentiment Treebank')
    print('=' * 80)

    # produce train/dev/test splits
    split(sst_dir, train_dir, dev_dir, test_dir)
    sent_paths = glob.glob(os.path.join(sst_dir, '*/sents.txt'))
    build_vocab(sent_paths, os.path.join(sst_dir, 'vocab-cased.txt'), lowercase=False)

    # write sentiment labels for nodes in trees
    dictionary = load_dictionary(sst_dir)
    write_labels(train_dir, dictionary)
    write_labels(dev_dir, dictionary)
    write_labels(test_dir, dictionary)

