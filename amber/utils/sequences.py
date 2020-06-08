# Author: Evan M. Cofer
# Created on June 7, 2020
"""
This module provides the `Sequence` class, which is an abstract class
that defines the interface for loading biological sequence data.
"""
import abc
import copy

import numpy
import pyfaidx


class Sequence(metaclass=abc.ABCMeta):
    """This class represents a source of sequence data, which can be
    fetched by querying different coordinates.
    """
    @abc.abstractmethod
    def __len__(self):
        """Number of queryable positions in the sequence.

        Returns
        -------
        int
            The number of queryable positions.

        """
        pass

    @abc.abstractmethod
    def coords_are_valid(self, *args, **kwargs):
        """Checks if the queried coordinates are valid.

        Returns
        -------
        bool
            `True` if the coordinates are valid, otherwise `False`.
        """
        pass

    @abc.abstractmethod
    def get_sequence_from_coords(self, *args, **kwargs):
        """Fetches a string representation of a sequence at
        the specified coordinates.

        Returns
        -------
        str
            The sequence of bases occuring at the queried
            coordinates. Behavior is undefined for invalid
            coordinates.
        """
        pass


class Genome(Sequence):
    """This class allows the user to a query a potentially file-backed
    genome by coordinate. It is essentially a wrapper around the
    `pyfaidx.Fasta` class.

    Parameters
    ----------
    input_path : str
        Path to an indexed FASTA file.
    in_memory : bool
        Specifies whether the genome should be loaded from
        disk and stored in memory.


    Attributes
    ----------
    data : pyfaidx.Fasta or dict
        The FASTA file containing the genome sequence. Alternatively,
        this can be a `dict` object mapping chromosomes to sequences that
        stores the file in memory.
    in_memory : bool
        Specified whether the genomic data is being stored in memory.
    chrom_len_dict : dict
        A dictionary mapping the chromosome names to their lengths.
    """
    def __init__(self, input_path, in_memory=False):
        """
        Constructs a new `Genome` object.
        """
        super(Genome, self).__init__()
        self.in_memory = in_memory
        if in_memory is True:
            fasta = pyfaidx.Fasta(input_path)
            self.data = {k: str(fasta[k][:].seq).upper() for k in fasta.keys()}
            fasta.close()
        else:
            self.data = pyfaidx.Fasta(input_path)
        self.chrom_len_dict = {k: len(self.data[k]) for k in self.data.keys()}

    def __del__(self):
        """
        Destructor method for `Genome` object.
        """
        if not self.in_memory:
            self.data.close()

    def __len__(self):
        """Number of queryable positions in the genome.

        Returns
        -------
        int
            The number of queryable positions.
        """
        return sum(self.chrom_len_dict.values())

    def coords_are_valid(self, chrom, start, end):
        """Checks if the queried coordinates are valid.

        Parameters
        ----------
        chrom : str
            The chromosome to query from.
        start : int
            The first position in the queried corodinates.
        end : int
            One past the last position in the queried coordinates.

        Returns
        -------
        bool
            `True` if the coordinates are valid, otherwise `False`.
        """
        if chrom not in self.chrom_len_dict:
            return False
        elif start < 0 or end <= 0:
            return False
        elif start >= end:
            return False
        elif end > self.chrom_len_dict[chrom]:
            return False
        elif start >= self.chrom_len_dict[chrom]:
            return False
        else:
            return True

    def get_sequence_from_coords(self, chrom, start, end):
        """Fetches a string representation of a sequence at
        the specified coordinates.


        Parameters
        ----------
        chrom : str
            Chromosome to query from.
        start : int
            First position in queried sequence.
        end : int
            One past the last position in the queried sequence.

        Returns
        -------
        str
            The sequence of bases occuring at the queried
            coordinates.

        Raises
        ------
        IndexError
            If the coordinates are not valid.
        """
        if self.coords_are_valid(chrom, start, end):
            x = self.data[chrom][start:end]
            if not self.in_memory:
                x = str(x.seq).upper()
            return x
        else:
            s = "Specified coordinates ({} to {} on \"{}\") are invalid!".format(
                start, end, chrom)
            raise IndexError(s)


class Encoding(metaclass=abc.ABCMeta):
    """This class is a mostly-abstract class used to represent some dataset that
    should be transformed with an encoding.
    """
    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        """
        Method to encode some input.
        """
        pass


class EncodedSequence(Encoding, Sequence):
    """Mixin of `Encoding` and `Sequence` to define the approach for
    encoding biological sequence data.
    """
    @property
    @abc.abstractmethod
    def ALPHABET_TO_ARRAY(self):
        """
        The alphabet used to encode the input sequence.
        """
        pass

    def encode(self, s):
        """Encodes a string with a numpy array.

        Parameters
        ----------
        s : str
            The string to encode.

        Returns
        -------
        numpy.ndarray
            An array with the encoded string.
        """
        ret = list()
        for i in range(len(s)):
            ret.append(self.ALPHABET_TO_ARRAY[s[i]])
        ret = numpy.stack(ret)
        return copy.deepcopy(ret)

    def get_sequence_from_coords(self, *args, **kwargs):
        """Fetches an encoded sequence at the specified coordinates.

        Returns
        -------
        numpy.ndarray
            The numpy array encoding the queried sequence.
        """
        return self.encode(
            super(EncodedSequence, self).get_sequence_from_coords(
                *args, **kwargs))


class EncodedGenome(EncodedSequence, Genome):
    """This class allows the user to a query a potentially file-backed
    genome by coordinate. It is essentially a wrapper around the
    `pyfaidx.Fasta` class. The returned values have been encoded as numpy
    arrays.

    Parameters
    ----------
    input_path : str
        Path to an indexed FASTA file.
    in_memory : bool
        Specifies whether the genome should be loaded from
        disk and stored in memory.


    Attributes
    ----------
    data : pyfaidx.Fasta or dict
        The FASTA file containing the genome sequence. Alternatively,
        this can be a `dict` object mapping chromosomes to sequences that
        stores the file in memory.
    in_memory : bool
        Specified whether the genomic data is being stored in memory.
    chrom_len_dict : dict
        A dictionary mapping the chromosome names to their lengths.
    ALPHABET_TO_ARRAY : dict
        A mapping from characters in the genome to their
        `numpy.ndarray` representations.
    """
    ALPHABET_TO_ARRAY = dict(A=numpy.array([1, 0, 0, 0]),
                             C=numpy.array([0, 1, 0, 0]),
                             G=numpy.array([0, 0, 1, 0]),
                             T=numpy.array([0, 0, 0, 1]),
                             N=numpy.array([.25, .25, .25, .25]))
    """
    A dictionary mapping possible characters in the genome to
    their `numpy.ndarray` representations.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructs a new `EncodedGenome` object.
        """
        super(EncodedGenome, self).__init__(*args, **kwargs)
