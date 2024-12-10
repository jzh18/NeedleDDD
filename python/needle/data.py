import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd
import gzip


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # BEGIN YOUR SOLUTION
    byte_order = 'big'
    X = []
    with gzip.open(image_filesname) as f:
        magic_num = int.from_bytes(f.read(4), byteorder=byte_order)
        if magic_num != 2051:
            raise Exception('decode error!')
        num_of_imgs = int.from_bytes(f.read(4), byteorder=byte_order)
        num_of_rows = int.from_bytes(f.read(4), byteorder=byte_order)
        num_of_cols = int.from_bytes(f.read(4), byteorder=byte_order)
        for i in range(num_of_imgs):
            img_one_dim = np.frombuffer(
                f.read(num_of_rows * num_of_cols), dtype=np.uint8).astype(np.float32)
            X.append(img_one_dim)
    X = np.array(X)
    min = np.min(X)
    max = np.max(X)
    X = (X - min) / (max - min)
    y = []
    with gzip.open(label_filename) as f:
        magic_num = int.from_bytes(f.read(4), byteorder=byte_order)
        if magic_num != 2049:
            raise Exception('decode error!')
        num_of_imgs = int.from_bytes(f.read(4), byteorder=byte_order)
        for i in range(num_of_imgs):
            label = np.frombuffer(f.read(1), dtype=np.uint8)
            y.append(label)

    return X, np.array(y).squeeze()

    # END YOUR SOLUTION


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        # BEGIN YOUR SOLUTION
        if flip_img:
            return np.fliplr(img)
        else:
            return img
        # END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device=None,
        dtype="float32"
    ):
        # for cuda test
        self.device = device
        self.dtype = dtype
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        # BEGIN YOUR SOLUTION
        if self.shuffle:
            all_indices = np.arange(len(self.dataset))
            np.random.shuffle(all_indices)
            self.ordering = np.array_split(all_indices, range(
                self.batch_size, len(self.dataset), self.batch_size))
        self.n = len(self.ordering)
        self.i = 0
        # END YOUR SOLUTION
        return self

    def __next__(self):
        # BEGIN YOUR SOLUTION
        if self.i >= self.n:
            raise StopIteration
        indices = self.ordering[self.i]
        self.i += 1

        all_res = []
        for i in indices:
            data = self.dataset[i]
            for j, v in enumerate(data):
                if j+1 > len(all_res):
                    all_res.append(list())
                all_res[j].append(v)

        results = []
        for v in all_res:
            t = Tensor(v, device=self.device, dtype=self.dtype)
            results.append(t)
        return results


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # BEGIN YOUR SOLUTION
        super().__init__(transforms)
        X, y = parse_mnist(image_filename, label_filename)
        size = len(X)
        self.X = X.reshape(size, 28, 28, 1)
        self.y = y
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # BEGIN YOUR SOLUTION
        x, y = self.X[index], self.y[index]
        x = self.apply_transforms(x)
        return x, y
        # END YOUR SOLUTION

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return len(self.X)
        # END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        # BEGIN YOUR SOLUTION
        super().__init__(transforms)
        if train:
            files = [f'data_batch_{x}' for x in range(1, 6)]
        else:
            files = ['test_batch']

        imgs = []
        labels = []
        for f in files:
            dict = self._unpickle(os.path.join(base_folder, f))
            imgs.append(dict[b'data'])
            labels.append(dict[b'labels'])
        self.X = np.concatenate(imgs, axis=0)/255.0
        self.y = np.concatenate(labels, axis=0)
        # END YOUR SOLUTION

    def _unpickle(self, file):

        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        # BEGIN YOUR SOLUTION
        return self.X[index].reshape(3, 32, 32), self.y[index]
        # END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        # BEGIN YOUR SOLUTION
        return len(self.y)
        # END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        # BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
        # END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        # BEGIN YOUR SOLUTION
        return len(self.idx2word)
        # END YOUR SOLUTION


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """

    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(
            base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(
            base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        # BEGIN YOUR SOLUTION
        tokens = []
        with open(path) as f:
            lines = f.read().splitlines()
            lines = lines[:max_lines]
            for l in lines:
                for w in l.split():
                    self.dictionary.add_word(w)
                self.dictionary.add_word('<eos>')
            for l in lines:
                for w in l.split():
                    tokens.append(self.dictionary.word2idx[w])
                tokens.append(self.dictionary.word2idx['<eos>'])
        return tokens

        # END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    # BEGIN YOUR SOLUTION
    size = len(data)
    size -= size % batch_size
    data = data[:size]
    data = np.array(data)
    h, w = batch_size, size//batch_size,
    data = data.reshape(h, w).T
    return data
    # END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    # BEGIN YOUR SOLUTION
    #print(f'batch size: {batches.shape[0]}, i: {i}, bptt: {bptt}')
    data = batches[i:i+bptt]
    target = batches[i+1:i+bptt+1]
   # print(f'target: {target}')
   # todo: should we transpose it or not
    #target = target.T
  #  print(f'target0: {target}')
    data = nd.array(data, device=device, dtype=dtype)
    target = nd.array(target, device=device, dtype=dtype).compact().reshape(
        (bptt*batches.shape[1],))
 #   print(f'target1: {target}')
    data = Tensor(data, device=device, dtype=dtype)
    target = Tensor(target, device=device, dtype=dtype)
#    print(f'target2: {target}')

    return data, target
    # return None
    # END YOUR SOLUTION
