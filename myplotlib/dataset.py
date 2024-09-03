from pandas import read_csv
from numpy import array

class BackendDataset:
  def __init__(self, name, samples, x_fn, x_label, y_fn, y_label, selector):
    self.name = name
    self.filter_x_fn = None
    read_all = [read_csv(s, sep="\t") for s in samples]
    self.samples = []
    for df in read_all:
      d = df
      for k,v in selector.items():
        d = d[d[k] == v]
      self.samples += [d]
    self.x_param = array([[x_fn(s)] for s in self.samples])
    self.y_param = array([[y_fn(s)] for s in self.samples])
    self.x_label = x_label
    self.y_label = y_label
    self.y_stack = {}
    