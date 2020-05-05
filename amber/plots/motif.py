# -*- coding: UTF-8 -*-

from __future__ import print_function

import gzip
from subprocess import *

import numpy as np

try:
    import matplotlib as mpl

    mpl.use('Agg')
    import seaborn
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-ticks')
    from matplotlib import transforms
    import matplotlib.patheffects
    from matplotlib.font_manager import FontProperties

    do_plot = True
except:
    do_plot = False


def read_file(filename):
    if filename.endswith('gz'):
        with gzip.GzipFile(filename) as fp:
            List = [x.strip().decode('utf-8') for x in fp if len(x.strip()) > 0]
    else:
        with open(filename) as fp:
            List = [x.strip() for x in fp if len(x.strip()) > 0]
    return List


def load_binding_motif_pssm(motif_file, is_log, swapbase=None, augment_rev_comp=False):
    dict_binding_pssm = {}
    rbp, pssm = '', []
    for line in read_file(motif_file):
        if line.startswith('#'):
            continue
        if line.startswith('>'):
            if len(pssm) > 0:
                if is_log:
                    pssm = convertlog2freq(pssm)
                pssm = np.array(pssm)
                if augment_rev_comp:
                    # print("augment rev comp")
                    pssm_rc = pssm[::-1, ::-1]
                if swapbase:
                    # print("swapbase: %s"%swapbase)
                    newbase = swapbase
                else:
                    newbase = [0, 1, 2, 3]
                dict_binding_pssm[rbp] = pssm[:, newbase]
                if augment_rev_comp:
                    dict_binding_pssm[rbp + '_rc'] = pssm_rc[:, newbase]
            rbp = line.strip()[1:].split()[0]
            pssm = []
        else:
            # pssm.append([float(x) for x in line.strip().split('\t')] + [0])
            ele = line.strip().split()
            column_prob = [float(x) for x in ele[-4:]]
            pssm.append(column_prob)
    return dict_binding_pssm


def convertlog2freq(pssm):
    pssm = np.array(pssm)
    pssm = np.power(2, pssm) * 0.25
    pssm = pssm[:, 0: 4].T
    # pssm = pssm - np.amin(pssm, axis = 0)
    pssm = pssm / np.sum(pssm, axis=0)
    return pssm.T


def draw_dnalogo_Rscript(pssm, savefn='seq_logo.pdf'):
    width = 4. / 8. * pssm.shape[1]
    pssm_flatten = pssm.flatten()
    seq_len = len(pssm[0])
    fw = open('/tmp/draw.seq.log.R', 'w')
    fw.write('seq_profile = c(' + ','.join([str(x) for x in pssm_flatten]) + ')' + '\n')
    fw.write('seq_matrix = matrix(seq_profile, 4, {}, byrow = T)'.format(seq_len) + '\n')
    fw.write("rownames(seq_matrix) <- c('A', 'C', 'G', 'T')" + '\n')
    fw.write("library(ggseqlogo)" + '\n')
    fw.write("library(ggplot2)" + '\n')
    fw.write("p <- ggplot() + geom_logo(seq_matrix) + theme_logo() + theme(axis.text.x = element_blank(), " + '\n')
    fw.write(
        "                                                                                                       panel.spacing = unit(0.5, 'lines')," + '\n')
    fw.write(
        "                                                                                                       axis.text.y = element_blank(), " + '\n')
    fw.write(
        "                                                                                                       axis.title.y = element_blank()," + '\n')
    fw.write(
        "                                                                                                       axis.title.x = element_blank(), " + '\n')
    fw.write(
        "                                                                                                       plot.title = element_text(hjust = 0.5, size = 20)," + '\n')
    fw.write(
        "                                                                                                       legend.position = 'none') + ggtitle('seqlogo')" + '\n')
    fw.write("ggsave('%s', units='in', width=%i, height=4 )\n" % (savefn, width))
    fw.close()
    cmd = 'Rscript /tmp/draw.seq.log.R'
    call(cmd, shell=True)


def draw_dnalogo_matplot(pssm):
    all_scores = []
    letter_idx = ['A', 'C', 'G', 'T']
    for row in pssm:
        tmp = []
        for i in range(4):
            tmp.append((letter_idx[i], row[i]))
        all_scores.append(sorted(tmp, key=lambda x: x[1]))
    draw_logo(all_scores)


class Scale(matplotlib.patheffects.RendererBase):
    '''http://nbviewer.jupyter.org/github/saketkc/notebooks/blob/master/python/Sequence%20Logo%20Python%20%20--%20Any%20font.ipynb?flush=true
    ## Author: Saket Choudhar [saketkc\\gmail]
    ## License: GPL v3
    ## Copyright © 2017 Saket Choudhary<saketkc__AT__gmail>
    '''

    def __init__(self, sx, sy=None):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy) + affine
        renderer.draw_path(gc, tpath, affine, rgbFace)


def draw_logo(all_scores, fontfamily='Arial', size=80,
              COLOR_SCHEME={'G': 'orange', 'A': 'red', 'C': 'blue', 'T': 'darkgreen'}):
    '''http://nbviewer.jupyter.org/github/saketkc/notebooks/blob/master/python/Sequence%20Logo%20Python%20%20--%20Any%20font.ipynb?flush=true
    ## Author: Saket Choudhar [saketkc\\gmail]
    ## License: GPL v3
    ## Copyright © 2017 Saket Choudhary<saketkc__AT__gmail>
    :: example:
    COLOR_SCHEME = {'G': 'orange', 
                'A': 'red', 
                'C': 'blue', 
                'T': 'darkgreen'}

BASES = list(COLOR_SCHEME.keys())


ALL_SCORES1 = [[('C', 0.02247014831444764),
              ('T', 0.057903843733384308),
              ('A', 0.10370837683591219),
              ('G', 0.24803586793255664)],
             [('T', 0.046608227674354567),
              ('G', 0.048827667087419063),
              ('A', 0.084338697696451109),
              ('C', 0.92994511407402669)],
             [('G', 0.0),
              ('T', 0.011098351287382456),
              ('A', 0.022196702574764911),
              ('C', 1.8164301607015951)],
             [('C', 0.020803153636453006),
              ('T', 0.078011826136698756),
              ('G', 0.11268374886412044),
              ('A', 0.65529933954826969)],
             [('T', 0.017393530660176126),
              ('A', 0.030438678655308221),
              ('G', 0.22611589858228964),
              ('C', 0.45078233627623127)],
             [('G', 0.022364103549245576),
              ('A', 0.043412671595594352),
              ('T', 0.097349627214363091),
              ('C', 0.1657574733649966)],
             [('C', 0.03264675899941203),
              ('T', 0.045203204768416654),
              ('G', 0.082872542075430544),
              ('A', 1.0949220710572034)],
             [('C', 0.0),
              ('T', 0.0076232429756614498),
              ('A', 0.011434864463492175),
              ('G', 1.8867526364762088)],
             [('C', 0.0018955903000026028),
              ('T', 0.0094779515000130137),
              ('A', 0.35637097640048931),
              ('G', 0.58005063180079641)],
             [('A', 0.01594690817903021),
              ('C', 0.017541598996933229),
              ('T', 0.2774762023151256),
              ('G', 0.48638069946042134)],
             [('A', 0.003770051401807444),
              ('C', 0.0075401028036148881),
              ('T', 0.011310154205422331),
              ('G', 1.8624053924928772)],
             [('C', 0.036479877757360731),
              ('A', 0.041691288865555121),
              ('T', 0.072959755514721461),
              ('G', 1.1517218549109602)],
             [('G', 0.011831087684038642),
              ('T', 0.068620308567424126),
              ('A', 0.10174735408273231),
              ('C', 1.0009100180696691)],
             [('C', 0.015871770937774379),
              ('T', 0.018757547471915176),
              ('A', 0.32176408355669878),
              ('G', 0.36505073156881074)],
             [('A', 0.022798100897300954),
              ('T', 0.024064662058262118),
              ('G', 0.24571286522646588),
              ('C', 0.34070495229855319)]]
    draw_logo(ALL_SCORES1, 'xkcd')
    '''
    if fontfamily == 'xkcd':
        plt.xkcd()
    else:
        mpl.rcParams['font.family'] = fontfamily

    fig, ax = plt.subplots(figsize=(len(all_scores), 2.5))

    font = FontProperties()
    font.set_size(size)
    font.set_weight('bold')

    # font.set_family(fontfamily)

    ax.set_xticks(range(1, len(all_scores) + 1))
    ax.set_yticks(range(0, 3))
    ax.set_xticklabels(range(1, len(all_scores) + 1), rotation=90)
    ax.set_yticklabels(np.arange(0, 3, 1))
    seaborn.despine(ax=ax, trim=True)

    trans_offset = transforms.offset_copy(ax.transData,
                                          fig=fig,
                                          x=1,
                                          y=0,
                                          units='dots')

    for index, scores in enumerate(all_scores):
        yshift = 0
        for base, score in scores:
            txt = ax.text(index + 1,
                          0,
                          base,
                          transform=trans_offset,
                          fontsize=80,
                          color=COLOR_SCHEME[base],
                          ha='center',
                          fontproperties=font,

                          )
            txt.set_path_effects([Scale(1.0, score)])
            fig.canvas.draw()
            window_ext = txt.get_window_extent(txt._renderer)
            yshift = window_ext.height * score
            trans_offset = transforms.offset_copy(txt._transform,
                                                  fig=fig,
                                                  y=yshift,
                                                  units='points')
        trans_offset = transforms.offset_copy(ax.transData,
                                              fig=fig,
                                              x=1,
                                              y=0,
                                              units='points')
    plt.show()
