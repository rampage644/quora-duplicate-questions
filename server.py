'''Server for visualization.'''
#%%
from __future__ import (absolute_import, print_function, unicode_literals, division)

import os
import json
import collections
import itertools
import numpy as np

from bokeh.layouts import column, gridplot, row
from bokeh.models.widgets import Button, MultiSelect
from bokeh.palettes import Set1
from bokeh.plotting import figure, curdoc, output_notebook, show



FILENAME = 'log'
COLORS = Set1[9]
STYLES = ['solid', 'dotted'] * 2
TIME_KEY = 'epoch'
WINDOW_SIZE = 20
DATA_KEYS = ['main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']
PATH = 'results/'

#%%
loss_plot = figure(plot_width=1000, plot_height=800)
lines = []


def discover():
    ret = []
    for root, _, files in os.walk(PATH):
        ret.extend(os.path.join(root, f) for f in files if 'log' in f)
    return sorted(ret, key=lambda x: os.path.getmtime(x), reverse=True)


def callback():
    dataseries = collections.defaultdict(dict)
    def get(key):
        return dataseries[key]


    for item in select.value:
        with open(item) as ifile:
            data = json.load(ifile)
            for key in DATA_KEYS:
                val = np.array([
                    (rcrd[TIME_KEY], rcrd[key]) for rcrd in data if key in rcrd], 'f')
                dataseries[item][key] = val
                if not len(dataseries[item][key]):
                    del dataseries[item][key]


    for idx, (title, data) in enumerate(dataseries.items()):
        train_loss = data.get('main/loss')
        test_loss = data.get('validation/main/loss')

        args = {
            'xs': [train_loss[:, 0], test_loss[:, 0]],
            'ys': [train_loss[:, 1], test_loss[:, 1]],
            'legend': title
        }
        loss_plot.multi_line(**args, line_color=COLORS[idx])


# # add a button widget and configure with the call back
button = Button(label="Update")
button.on_click(callback)

select = MultiSelect(title='Results',
                     size=40,
                     options=discover())

# output_notebook()
layout = row(column(select, button), loss_plot)
curdoc().add_root(layout)


