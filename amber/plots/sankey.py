"""
Sankey plot for visualizing architecture transitioning probabilities

Based on Evan Cofer's original notebook: https://github.com/zj-zhang/AMBER/blob/48db1d3adc9988ded4767aec0796c908c09f0085/examples/AMBER_analyze_third_search.ipynb
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import plotly
from ..architect.modelSpace import get_layer_shortname

DEFAULT_PALETTE = ['tomato', 'cornflowerblue', 'gold', 'orchid', 'seagreen', 'cyan', 'grey']


def get_random_palette(number_of_colors, seed=None):
    np.random.seed(seed)
    color = ["#" + ''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)])
             for i in range(number_of_colors)]
    # print(color)
    return color


def plot_grouped_sankey(actions_dict, model_space, save_fn, palette=DEFAULT_PALETTE):
    """grouped sankey plot where color ribbons represent different architectures

    """
    data_dict = {"sources": [],
                 "targets": [],
                 "values": [],
                 "labels": [],
                 "colors": []}
    palette = palette or get_random_palette(len(actions_dict))
    assert len(palette) >= len(actions_dict)
    for i, (group_id, group) in enumerate(actions_dict.items()):
        # Convert to array.
        layer_cols = [x for x in list(group.columns) if x.startswith("layer_")]
        vals = group[layer_cols].values

        # Convert model outputs to sankey inputs.
        sources, targets, values, labels = get_sankey_diagram_inputs(vals, model_space)
        data_dict["sources"] += sources
        data_dict["targets"] += targets
        data_dict["values"] += values
        data_dict["labels"] += labels
        data_dict["colors"] += [palette[i]] * len(targets)
        print(group_id, palette[i])

    # Convert labels to model layer info.
    label_strs = list()
    for i in range(len(model_space)):
        for j in range(len(model_space[i])):
            x = model_space[i][j]
            s = get_layer_shortname(x)
            label_strs.append(s)

    # Build plot.
    fig = plotly.graph_objects.Figure(data=[plotly.graph_objects.Sankey(
        node=dict(
            label=label_strs,
            color="lightgrey"
        ),
        textfont=dict(size=10),
        link=dict(
            source=data_dict["sources"],
            target=data_dict["targets"],
            value=data_dict["values"],
            color=data_dict["colors"],
        ))])
    fig.update_layout(width=1200, height=500, showlegend=True)
    fig.write_image(save_fn)


def plot_sankey(controller, model_space, save_fn, B=100, get_kwargs=None, palette=DEFAULT_PALETTE):
    """Plot the sankey diagram to represent the transition patterns, by sampling from controller

    Parameters
    ----------
    controller : amber.architect.GeneralController
        a controller instance that implements "get_action" method
    model_space : amber.architect.modelSpace
        model space for labeling layers
    save_fn : str
        file path for saving
    B : int
        number of samples to draw from controller
    get_kwargs : dict
        keyword arguments to parse to `controller.get_action`
    palette : list
        list of colors

    Returns
    ----------
    pd.DataFrame : a dataframe of sampled architectures

    """
    if palette is None or len(palette) < model_space.get_space_size():
        palette = get_random_palette(number_of_colors=20, seed=777)
    get_kwargs = get_kwargs or {}
    # Sample actions from trained controller
    actions = []
    for _ in range(B):
        actions.append(
            controller.get_action(**get_kwargs)[0])  # Returns one-hot and probabilities.
    actions = np.array(actions)
    sources, targets, values, labels = get_sankey_diagram_inputs(actions, model_space)
    # Mapping from layer kinds to colors.
    cmap_list = list()
    for i in range(len(model_space)):
        cmap_list.append(dict())
        for j in range(len(model_space[i])):
            s = get_layer_shortname(model_space[i][j])
            cmap_list[-1][s] = palette[j]

    # Convert labels to model layer info.
    label_strs = list()
    for i, j in labels:
        x = model_space[i][j]
        s = get_layer_shortname(x)
        label_strs.append(s)
    #     label_strs.append("{} - {}".format(x.Layer_type, x.Layer_attributes))

    # Get colors for model.
    colors = list()
    for (i, _), s in zip(labels, label_strs):
        colors.append(cmap_list[i][s])

    # Build plot.
    fig = plotly.graph_objects.Figure(data=[plotly.graph_objects.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label_strs,
            color=colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ))])
    fig.write_image(save_fn)
    actions_df = pd.DataFrame(actions, columns=["layer_{}".format(i) for i in range(actions.shape[1])])
    return actions_df


def get_sankey_diagram_inputs(m, model_space):
    """
    Make a sankey plot for a set of model configs.

    Parameters
    ----------
    m : np.array
        Rows are records. Each column is a level in the graph.
    model_space : amber.architect.modelSpace
        model space for consistent token lookup

    """
    # Change tokens to make lookup easier.
    token_lookup = dict()
    for i in range(len(model_space)):
        for j in range(len(model_space[i])):
            token_lookup[(i, j)] = len(token_lookup)
    # Convert matrix.
    mm = np.zeros_like(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            mm[i, j] = token_lookup[(j, m[i, j])]
    #     print(mm)
    # Create connections etc.
    g = defaultdict(lambda: 0)
    for i in range(mm.shape[0]):
        for j in range(1, mm.shape[1]):
            g[(mm[i, j - 1], mm[i, j])] += 1
    # Format args for sankey plot.
    sources = list()
    targets = list()
    values = list()
    for (src, snk), v in g.items():
        sources.append(src)
        targets.append(snk)
        values.append(v)
    # Get labels for plot.
    labels = list()
    for k, v in sorted(token_lookup.items(), key=lambda x: x[1]):
        labels.append(k)

    # Finished.
    return sources, targets, values, labels
