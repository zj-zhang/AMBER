"""
Evan Cofer, 2020
"""
import argparse
import collections
import os
import re

import intervaltree
import gzip
import numpy
import pyfaidx

def draw_samples(genome_file, bed_file, output_file, feature_name_file,
                 bin_size, cvg_frac, n_examples,
                 chrom_pad, chrom_pattern, max_unk):
    # Read feature names from file.
    feature_name_set = set()
    i_to_feature_name = list()
    feature_name_to_i = dict()
    i = 0
    with open(feature_name_file, "r") as read_file:
        for line in read_file:
            line = line.strip()
            if line:
                feature_name_set.add(line)
                i_to_feature_name.append(line)
                feature_name_to_i[line] = i
                i += 1
    n_feats = len(i_to_feature_name)

    # Load genome and get estimate of chromosome weights.
    genome = pyfaidx.Fasta(genome_file)
    chroms = list()
    chrom_lens = list()
    max_examples = 0
    for k in genome.keys():
        if chrom_pattern.match(k) is not None:
            l = len(genome[k])
            if l > chrom_pad * 3:
                chroms.append(k)
                chrom_lens.append(l - 2 * chrom_pad - bin_size)
    chrom_weighting_lens = numpy.array(chrom_lens)
    chrom_lens = numpy.array(chrom_lens) * 2
    chrom_weights = chrom_lens / numpy.sum(chrom_lens)
    n_chrom = len(chroms)
    max_examples = numpy.sum(chrom_lens)
    if max_examples < n_examples:
        msg = "Got {} max examples possible, but need {} examples".format(
            max_examples, n_examples)
        raise ValueError(msg)

    # Create interval tree for fast label query.
    ivt = {"+": collections.defaultdict(intervaltree.IntervalTree),
           "-": collections.defaultdict(intervaltree.IntervalTree)}
    with gzip.open(bed_file, "rt") as read_file:
        for line in read_file:
            line = line.strip()
            if not line.startswith("#"):
                line = line.split("\t")
                chrom, start, end, name = line[:4]
                if len(line) >= 6:
                    strand = [line[5]]
                else:
                    strand = ["+", "-"]
                start = int(start)
                end = int(end)
                if name in feature_name_set:
                    for x in strand:
                        ivt[x][chrom].addi(start, end, feature_name_to_i[name])

    # Create outputs.
    seen = {"+": collections.defaultdict(set),
            "-": collections.defaultdict(set)}
    outputs = list()
    i = 0
    while i < n_examples:
        c_i = numpy.random.choice(n_chrom, p=chrom_weights)
        strand = numpy.random.choice(["+", "-"], 1)[0]
        chrom = chroms[c_i]
        pos = numpy.random.choice(chrom_lens[c_i]) + chrom_pad
        start = pos
        end = pos + bin_size
        if pos not in seen[strand][c_i]:
            # Add to seen and adjust weights.
            chrom_weighting_lens[c_i] -= 1
            chrom_weights = chrom_weighting_lens / numpy.sum(chrom_weighting_lens)
            seen[strand][c_i].add(pos)

            # Determine label etc w/ ivt.
            cvg = numpy.zeros(n_feats)
            for x in ivt[strand][chroms[c_i]].overlap(start, end):
                cvg[x.data] += min(x.end, end) - max(x.begin, start)
            cvg /= bin_size
            cvg = (cvg > cvg_frac).astype(int).tolist()
            outputs.append((chrom, start, end, strand, *cvg))
            i += 1

    # write outputs to file.
    with open(output_file, "w") as write_file:
        for x in sorted(outputs):
            x = [str(y) for y in list(x)]
            write_file.write("\t".join(x) + "\n")


if __name__ == "__main__":
    # Get command line arguments.
    parser = argparse.ArgumentParser( description="sampling data for tf genomics models")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--genome", type=str, required=True, help="Path to the indexed fasta file.")
    parser.add_argument("--bed", type=str, required=True, help="Path to the input bed file.")
    parser.add_argument("--feature-name-file", type=str, required=True, help="Name of feature in bed file to look for.")
    parser.add_argument("--bin-size", type=int, required=True, help="Size of the bin composing minimal examples.")
    parser.add_argument("--cvg-frac", type=float, required=True, help="Fraction of bin that must be covered to be positive example.")
    parser.add_argument("--n-examples", type=int, required=True, help="Number of examples to include.")
    parser.add_argument("--chrom-pad", type=int, required=True, help="Length of region to ignore at the start and end of chromosomes.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for RNG.")
    parser.add_argument("--include-chroms", type=str, required=True, help="Regex for chromosomes to include.")
    parser.add_argument("--max-n", type=int, required=True, help="Maximum N chars in sequences.")
    args = parser.parse_args()

    # Validate arguments.
    for x in [args.genome, args.bed, args.feature_name_file]:
        if not os.path.exists(x):
            raise ValueError(x + " does not exist")

    if args.n_examples <= 0:
        raise ValueError("--n-examples must be > 0")

    if args.chrom_pad < 0:
        raise ValueError("--chrom-pad must be >= 0")

    if args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")

    if args.cvg_frac < 0:
        raise ValueError("--cvg-frac must be >= 0")
    elif args.cvg_frac > 1:
        raise ValueError("--cvg-frac must be < 1")

    if args.max_n < 0:
        raise Valueerror("--max-n must be >= 0")


    # Prepare for function.
    numpy.random.seed(args.seed)
    pattern = re.compile("^" + args.include_chroms + "$")

    # Run function.
    draw_samples(args.genome, args.bed, args.output, args.feature_name_file,
                 args.bin_size, args.cvg_frac, args.n_examples,
                 args.chrom_pad, pattern, args.max_n)
