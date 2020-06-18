# Author: Evan M. Cofer
# Created on June 5, 2020
"""
This module provides the `BioIntervalSource` class and its children.
These are essentially wrappers for sets of sequence intervals and
associated labels.
"""
import keras
import numpy


class BioIntervalSource(object):
    """A generic class for labeled examples of biological intervals.
    The amount of padding added to the end of the intervals is able to
    be changed during runtime. This allows these functions to be passed
    to objects such as a model controller.

    Parameters
    ----------
    example_file : str
        A path to a file that contains the examples in BED-like format.
        Specifically, this file will have one example per line, with
        the chromosome, start, end, and label for the example. Each
        column is separated by tabs.
    reference_sequence : Sequence
        The reference sequence used to generate the input sequences
        from the example coordinates.

    Attributes
    ----------
    reference_sequence : Sequence
        The reference sequence used to generate the input sequences
        from the example coordinates.
    examples : list
        A list of the example coordinates.
    labels : list
        A list of the labels for the examples.
    left_pad : int
        The length of padding added to the left side of the interval.
    right_pad : int
        The length of padding added to the right side of the interval.
    """
    def __init__(self, example_file, reference_sequence):
        self.reference_sequence = reference_sequence
        self.left_pad = 0
        self.right_pad = 0

        # Load examples.
        self.labels = list()
        self.examples = list()
        with open(example_file, "r") as read_file:
            for line in read_file:
                line = line.strip()
                if not line.startswith("#"):
                    if line:
                        line = line.split("\t", 4)
                        chrom, start, end, strand = line[:4]
                        label = [int(x) for x in line[5:]]
                        self.labels.append(numpy.array([label]))
                        self.examples.append((chrom, int(start), int(end), strand))

    def padding_is_valid(self, value):
        """Determine if the specified value is a valid value for padding
        intervals.

        Parameters
        ----------
        value : int
            Proposed amount of padding.

        Returns
        -------
        bool
            Whether the input value is valid.
        """
        if value < 0:
            return False
        else:
            return True

    def _test_padding(self, value):
        """Tests if padding is valid or not. If invalid, raises an error.


        Parameters
        ----------
        value : int
            Amount of padding to test.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            This method throws an error if the proposed amount of padding is
            not a valid amount.
        """
        if not self.padding_is_valid(value):
            s = "Invalid padding amount : {}".format(value)
            raise ValueError(s)

    def set_left_pad(self, value):
        """Sets the length of the padding added to the left
        side of the input sequence.

        Parameters
        ----------
        value : int
            The length of the padding to add to the left side of an example
            interval.
        """
        self._test_padding(value)
        self.left_pad = value

    def set_right_pad(self, value):
        """Sets the length of the padding added to the right side of an
        example interval.


        Parameters
        ----------
        value : int
            The length of the padding to add to the right side of an example
            interval.
        """
        self._test_padding(value)
        self.right_pad = value

    def set_pad(self, value):
        """Sets the length of padding added to both the left and right sides of
        example intervals.

        Parameters
        ----------
        value : int
            The length of the padding to add to the left and right sides of
            input example intervals.
        """
        self._test_padding(value)
        self.left_pad = value
        self.right_pad = value

    def __len__(self):
        """Number of examples available.

        Returns
        -------
        int
            The number of examples available.
        """
        return len(self.examples)

    def _load_unshuffled(self, item):
        """Loads example `item` from the unshuffled list of examples.

        Parameters
        ----------
        item : int
            The index of the example to load.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
        """
        chrom, start, end, strand = self.examples[item]
        x = self.reference_sequence.get_sequence_from_coords(chrom, start - self.left_pad, end + self.right_pad, strand)
        y = self.labels[item]
        return x, y


class BioIntervalSequence(BioIntervalSource, keras.utils.Sequence):
    """This data sequence type holds intervals in a genome and a
    label associated with each interval. Unlike a generator, this
    is based off of `keras.utils.Sequence`, which shifts things like
    shuffling elsewhere. The amount of padding added to the end of
    the intervals is able to be changed during runtime. This allows
    these functions to be passed to objects such as a model
    controller.

    Parameters
    ----------
    example_file : str
        A path to a file that contains the examples in BED-like format.
        Specifically, this file will have one example per line, with
        the chromosome, start, end, and label for the example. Each
        column is separated by tabs.
    reference_sequence : Sequence
        The reference sequence used to generate the input sequences
        from the example coordinates.

    Attributes
    ----------
    reference_sequence : Sequence
        The reference sequence used to generate the input sequences
        from the example coordinates.
    examples : list
        A list of the example coordinates.
    labels : list
        A list of the labels for the examples.
    left_pad : int
        The length of padding added to the left side of the interval.
    right_pad : int
        The length of padding added to the right side of the interval.
    """
    def __init__(self, example_file, reference_sequence):
        super(BioIntervalSequence, self).__init__(
            example_file, reference_sequence)

    def __getitem__(self, item):
        """
        Indexes into the set of examples and labels.

        Parameters
        ----------
        item : int
            The index in the example/label pairs to fetch.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            A tuple consisting of the example and the target label.

        """
        return self._load_unshuffled(item)


class BioIntervalGenerator(BioIntervalSource):
    """This data generator type holds intervals in a genome and a
    label associated with each interval. This essentially acts as
    an iterator over the inputs examples. This approach is useful
    and preferable to `BioIntervalSequence` when there are a very
    large number of examples in the input. The amount of padding
    added to the end of the intervals is able to be changed during
    runtime. This allows these functions to be passed to objects
    such as a model controller.

    Parameters
    ----------
    example_file : str
        A path to a file that contains the examples in BED-like format.
        Specifically, this file will have one example per line, with
        the chromosome, start, end, and label for the example. Each
        column is separated by tabs.
    reference_sequence : Sequence
        The reference sequence used to generate the input sequences
        from the example coordinates.

    Attributes
    ----------
    reference_sequence : Sequence
        The reference sequence used to generate the input sequences
        from the example coordinates.
    examples : list
        A list of the example coordinates.
    labels : list
        A list of the labels for the examples.
    left_pad : int
        The length of padding added to the left side of the interval.
    right_pad : int
        The length of padding added to the right side of the interval.
    """
    def __init__(self, example_file, reference_sequence):
        super(BioIntervalGenerator, self).__init__(
            example_file, reference_sequence)
        raise NotImplementedError


class BatchedBioIntervalSequence(BioIntervalSource, keras.utils.Sequence):
    """This data sequence type holds intervals in a genome and a
    label associated with each interval. Unlike a generator, this
    is based off of `keras.utils.Sequence`, which shifts things like
    shuffling elsewhere. The amount of padding added to the end of
    the intervals is able to be changed during runtime. This allows
    these functions to be passed to objects such as a model
    controller. Examples are divided into batches.

    Parameters
    ----------
    example_file : str
        A path to a file that contains the examples in BED-like format.
        Specifically, this file will have one example per line, with
        the chromosome, start, end, and label for the example. Each
        column is separated by tabs.
    reference_sequence : Sequence
        The reference sequence used to generate the input sequences
        from the example coordinates.
    batch_size : int
        Specifies size of the mini-batches.
    shuffle : bool
        Specifies whether to shuffle the mini-batches.
    seed : int
        Value to seed random number generator with.


    Attributes
    ----------
    reference_sequence : Sequence
        The reference sequence used to generate the input sequences
        from the example coordinates.
    examples : list
        A list of the example coordinates.
    labels : list
        A list of the labels for the examples.
    left_pad : int
        The length of padding added to the left side of the interval.
    right_pad : int
        The length of padding added to the right side of the interval.
    batch_size : int
        Specifies size of the mini-batches.
    shuffle : bool
        Specifies whether to shuffle the mini-batches.
    seed : int
        Value to seed random number generator with.
    random_state : numpy.random.RandomState
        A random number generator to use.
    """
    def __init__(self, example_file, reference_sequence,
                 batch_size, seed=1337, shuffle=True):
        super(BatchedBioIntervalSequence, self).__init__(
            example_file, reference_sequence)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.random_state = numpy.random.RandomState(seed=self.seed)
        self.index = numpy.arange(len(self.examples))

    def __len__(self):
        """Number of examples available.

        Returns
        -------
        int
            The number of examples available.
        """
        l = super(BatchedBioIntervalSequence, self).__len__()
        return l // self.batch_size

    def __getitem__(self, item):
        """
        Indexes into the set of examples and labels.

        Parameters
        ----------
        item : int
            The index in the example/label pairs to fetch.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            A tuple consisting of the example and the target label.

        """
        x = list()
        y = list()
        for i in range(self.batch_size):
            cur_x, cur_y = self._load_unshuffled(self.index[item + i])
            x.append(cur_x)
            y.append(cur_y)
        x = numpy.stack(x)
        y = numpy.stack(y)
        return x, y

    def on_epoch_end(self):
        """
        If applicable, shuffle the examples at the end of an epoch.
        """
        if self.shuffle:
            self.index = self.random_state.choice(len(self.examples),
                                                  len(self.examples),
                                                  replace=False)

