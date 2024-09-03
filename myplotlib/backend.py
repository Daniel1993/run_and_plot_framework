import numpy as np
import pandas as pd

class BackendDataset:
	def __init__(self, name, samples, selector):
		self.name = name
		self.filter_x_fn = None
		read_all = [pd.read_csv(s, sep="\t") for s in samples]
		self.samples = []
		self.opt = []
		for df in read_all:
			d = df
			for k,v in selector.items():
				if type(v) is list:
					d = d[d[k].isin(v)]
				else:
					d = d[d[k] == v]
			self.samples += [d]

	def add_line(self, opt):
		opt.add_backend(self)
		# self.line_opt += [opt]
		self.opt += [opt]
		return opt

	def add_stack(self, opt):
		y_stack = {}
		for lbl,fn in opt.y_fn.items():
			y_stack[lbl] = np.array([[fn(s)] for s in self.samples])
		opt.add_backend(self)
		# opt.set_y_stack(self, y_stack)
		# self.stack_opt += [opt]
		self.opt += [opt]
		return opt
	
	def plot(self):
		for o in self.opt:
			self.opt.plot()