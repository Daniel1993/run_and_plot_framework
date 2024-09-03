import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from typing import Callable, Optional, Tuple
import matplotlib.ticker as mtick
import json
import re
from myplotlib.backend import BackendDataset
from myplotlib.backend import BackendDataset
from typing import Callable
import numpy as np

PATTERNS = [
	"//" , "\\\\" , "||" , "+++" , "xxx", "xx", "///" , "\\\\\\" , "|||" , "++", "///" , "\\\\\\" , "|||", "//" , "\\\\" , "||"
]

MARKERS = [
	",", "o", "s", "*", "+", "x", "d", "v", "^", "h", "."
]

COLORS_BG_WT = [ # text better in white
	"#35B700", "#FC9C3A", "#DA3033", "#043B7D", "#8927B0", "#F53D96", "#DCC518", "#0B86F9"
]

COLORS_BG_BL = [ # text better in black
	"#BDD372", "#FECD8F", "#FC8CA1", "#8ED7D8", "#E78FFF", "#E28865", "#C8C8C8", "#16D0B1"
]

class PlotOptions:
	def __init__(self, title, x_label, y_label, figsize=(8,4)):
		self.title = title
		self.x_label = x_label
		self.y_label = y_label
		self.figsize = figsize
		self.legend_show = True
		self.backends = {}
		self.colors = {"___p1": COLORS_BG_WT, "___p2": COLORS_BG_BL}
		self.markers = {"___m1": MARKERS}
		self.patterns = {"___p1": PATTERNS}
		self.set_margins_offset()
	def set_xaxis(self, x_fn):
		self.x_fn = x_fn
		return self
	def add_backend(self, backend:BackendDataset):
		if backend not in self.backends:
			self.backends[backend] = None
		return self
	def adjust_legend(self, size = 10, show_legend = True):
		self.legend_size = size
		self.legend_show = show_legend
		mpl.rcParams.update({
			'legend.fontsize': size
		})
		return self
	def adjust_title(self, size = 'x-large', pad = 4.0):
		self.title_size = size
		self.title_padding = pad
		mpl.rcParams.update({
			'axes.titlesize': size,
			'axes.titlepad': pad,
		})
		return self
	def adjust_labels(self, size = "medium", pad = 4.0):
		self.label_size = size
		self.labelpad = pad
		mpl.rcParams.update({
			'axes.labelsize': size,
			'axes.labelpad': pad
		})
		return self
	def adjust_xticks(self, size = "medium"):
		self.xtick_size = size
		mpl.rcParams.update({
			'xtick.labelsize': size,
		})
		return self
	def adjust_yticks(self, size = "medium"):
		self.ytick_size = size
		mpl.rcParams.update({
			'ytick.labelsize': size,
		})
		return self
	def adjust_ylabel(self, size = 0, xpad = 0, ypad = 0):
		return self
	def add_labeled_color(self, label, color):
		self.colors[label] = color
		return self
	def prepare_colors(self):
		pass
	def set_margins_offset(self, hspace=0.0, wspace = 0.0, left=0.0, right=0.0, top=0.0, bottom=0.0):
		self.hspace = hspace
		self.wspace = wspace
		self.left   = left
		self.right  = right
		self.top    = top
		self.bottom = bottom
		return self
	def adjust_layout(self):
		"""
		Called by the plot script.
		"""
		plt.tight_layout()
		plt.subplots_adjust(
			hspace = mpl.rcParams["figure.subplot.hspace"] + self.hspace,
			wspace = mpl.rcParams["figure.subplot.wspace"] + self.wspace,
			left   = mpl.rcParams["figure.subplot.left"]   + self.left,
			right  = mpl.rcParams["figure.subplot.right"]  - self.right,
			top    = mpl.rcParams["figure.subplot.top"]    - self.top,
			bottom = mpl.rcParams["figure.subplot.bottom"] + self.bottom
		)
		return self
	def adjust_axis(self, ax):
		"""
		Called by the plot script.
		"""
		if self.legend_show:
			ax.legend()
		return self
	def prepare(self):
		self.x_params = {}
		self.y_params = {}
		for b in self.backends:
			self.x_params[b.name] = np.array([[self.x_fn(s)] for s in b.samples])
			if type(self.y_fn) == dict:
				self.y_params[b.name] = {}
				for l in self.y_fn:
					self.y_params[b.name][l] = np.array([[self.y_fn[l](s)] for s in b.samples])
			else:
				self.y_params[b.name] = np.array([[self.y_fn(s)] for s in b.samples])
		self.prepare_colors()
		self.adjust_layout()
	def plot(self):
		pass

class LineOptions(PlotOptions):
	def set_yaxis(self, y_fn):
		self.y_fn = y_fn
		self.y_stack = {}
		return self
	# def prepare(self):
	# 	super().prepare()
	def plot(self):
		LinesPlot(self).plot()
		
	def prepare_colors(self):
		idx = 0
		for b in self.backends:
			if not b.name in self.colors:
				l = len(self.colors["___p1"])
				self.markers[b.name] = self.markers["___m1"][idx%len(self.markers["___m1"])]
				if idx < l:
					self.colors[b.name] = self.colors["___p1"][idx]
				elif idx < l + len(self.colors["___p2"]):
					self.colors[b.name] = self.colors["___p2"][idx-l]
				else:
					raise Exception("Not enough colors")
				idx += 1

class StackOptions(PlotOptions):
	class y_FnBuilder:
		def __init__(self):
			self.fns = {}
			self.y_stack = None
		def b(self, bar_slice_name:str, fn:Callable):
			self.fns[bar_slice_name] = fn
			return self
	def __init__(self, title, x_label, y_label, y_fns:y_FnBuilder, **kwargs):
		"""
		You can add an optional function filter_x_fn((x,y,std_dev))->bool to discard points (when returning False).
		"""
		# defaults
		super().__init__(title, x_label, y_label)
		self.opt = {}
		self.title = title
		self.y_label = y_label
		self.y_fn = y_fns.fns
		self.opt["filter_x_fn"] = None # Optional[Callable[[Tuple[float,float,float]],bool]]
		self.opt["is_percent"] = False
		self.opt["fix_at_100_percent"] = False
		self.opt["top_extra"] = 0
		self.opt["filter_out_backends"] = []
		self.opt["break_y_scale"] = None
		self.opt["overide_max_y"] = None
		for key, value in kwargs.items():
			self.opt[key] = value
	# def set_y_stack(self, backend, y_stack:dict[str,np.ndarray]):
	# 	if self.backends == None:
	# 		raise Exception("backend does not exist")
	# 	if self.backends[backend] != None:
	# 		raise Exception("y_stack is already defined")
	# 	self.backends[backend] = {"y_stack": y_stack}
	def prepare_colors(self):
		idx = 0
		for b in self.backends:
			y_stack = self.backends[b]["y_stack"]
			for lbl in y_stack:
				if not lbl in self.colors:
					l = len(self.colors["___p1"])
					self.markers[lbl] = self.markers["___m1"][idx%len(self.markers["___m1"])]
					self.patterns[lbl] = self.patterns["___p1"][idx%len(self.patterns["___p1"])]
					if idx < l:
						self.colors[lbl] = self.colors["___p1"][idx]
					elif idx < l + len(self.colors["___p2"]):
						self.colors[lbl] = self.colors["___p2"][idx-l]
					else:
						raise Exception("Not enough colors")
					idx += 1
			break
	def add_backend(self, backend: BackendDataset):
		super().add_backend(backend)
		y_stack = {}
		x_stack = np.array([[self.x_fn(s)] for s in backend.samples])
		for lbl,fn in self.y_fn.items():
			y_stack[lbl] = np.array([[fn(s)] for s in backend.samples])
		self.backends[backend] = {"y_stack": y_stack, "x_stack": x_stack}
		return super().add_backend(backend)
	def plot(self):
		StackPlot(self).plot()

class Plot:
	def __init__(self, opt:PlotOptions):
		self.opt = opt
			
	def gather_stats(self):
		p = self.opt
		self.data_y_range = [{}, {}]
		self.data_x_range = [{}, {}]
		self.data_y_plus_std_range = [{}, {}]
		self.data_backend = {}
		for d in p.backends:
			y_params = {"y": p.y_params[d.name]}
			self.data_backend[d.name] = {}
			self.data_x_range[0][d.name] = {}
			self.data_x_range[1][d.name] = {}
			self.data_y_range[0][d.name] = {}
			self.data_y_range[1][d.name] = {}
			self.data_y_plus_std_range[0][d.name] = {}
			self.data_y_plus_std_range[1][d.name] = {}
			if type(p.y_params[d.name]) == dict:
				y_params = p.y_params[d.name]
			for lbl in y_params:
				z = zip(p.x_params[d.name].transpose(), y_params[lbl].transpose())
				t = [(np.average(x),np.average(y),np.std(y)) for x,y in z]
				t.sort(key=lambda elem : elem[0]) # sort by X
				self.data_backend[d.name][lbl] = [t]
				l = [(s[0], s[1], s[1]+s[2]) for s in t]
				m = [min(list(t)) for t in zip(*l)]
				self.data_x_range[0][d.name][lbl] = m[0]
				self.data_y_range[0][d.name][lbl] = m[1]
				self.data_y_plus_std_range[0][d.name][lbl] = m[2]
				M = [max(list(t)) for t in zip(*l)]
				self.data_x_range[1][d.name][lbl] = M[0]
				self.data_y_range[1][d.name][lbl] = M[1]
				self.data_y_plus_std_range[1][d.name][lbl] = M[2]

	def plot(self):
		"""
		Generates a .pdf with the plot.
		"""
		p = self.opt
		p.prepare()
		t = re.sub(r'[\W]', '_', p.title)
		self.filename = f"{t}.pdf"
		self.gather_stats()

class LinesPlot(Plot):
	def plot(self):
		p = self.opt
		fig, ax = plt.subplots(figsize=p.figsize, nrows=1, ncols=1)
		super().plot() # needs to come after the plt.subplots, as it sets the borders

		print(f"plotting \"{p.title}\"")

		for i,d in enumerate(p.backends):
			t = self.data_backend[d.name]
			x_array, y_array, y_error = zip(*t['y'][0])
			# breakpoint()
			color = p.colors[d.name]
			ax.errorbar(x_array, y_array, yerr = y_error, label=d.name, marker=p.markers[d.name], color=color)

		ax.set_xlabel(p.x_label, size=16)
		ax.set_ylabel(p.y_label, size=16)
		ax.set_title(p.title)
		# ax.legend(prop={'size': 14})
		max_y = max([self.data_y_range[1][bck]['y'] for bck in self.data_y_range[1]])
		ax.set_ylim(bottom=0, top=max_y + 0.04*max_y)
		# ax.tick_params(axis='both', which='major', labelsize=14)
		# ax.set_xticks([1, 4, 8, 16, 24, 32])
		# left_margin = 0.06 if max_y < 9.9 else 0.105
		
		p.adjust_axis(ax)
		plt.savefig(self.filename)
		fig.clear()
		plt.close()
		return self

class StackPlot(Plot):

	def plot(self):
		p = self.opt
		fig, ax = plt.subplots(figsize=p.figsize, nrows=1, ncols=1)
		axs = ax
		axs_all = None
		# breakpoint()
		if p.opt["break_y_scale"] != None:
			fig, axs_all = plt.subplots(figsize=p.figsize, nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 1]})
			axs = axs_all[0]
			axs_all[0].margins(x=0)
			axs_all[1].margins(x=0)
			axs_all[0].spines.bottom.set_visible(False)
			axs_all[0].xaxis.tick_top()
			# axs_all[0].spines.top.set_visible(False)
			axs_all[1].spines.top.set_visible(False)
			axs_all[0].tick_params(labeltop=False)
			kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=1,
										linestyle="dotted", color='k', mec='k', mew=0.01, clip_on=False)
			break_y_offset = 0
			break_y = p.opt["overide_max_y"] if not p.opt["overide_max_y"] is None else max([self.data_y_range[1][lbl] for lbl in self.data_y_range[1]])
			# TODO: check the index in axs_all[i+1] and axs_all[i] math is boggus
			axs_all[1].set_ylim([None,p.opt["break_y_scale"]]) # below
			axs_all[0].set_ylim([p.opt["break_y_scale"]+break_y_offset,break_y]) # top

			axs_all[0].plot([-0.1,4.75], [p.opt["break_y_scale"],p.opt["break_y_scale"]], **kwargs)
			axs_all[1].plot([-0.1,4.75], [p.opt["break_y_scale"],p.opt["break_y_scale"]], **kwargs)
		super().plot() # needs to come after the plt.subplots, as it sets the borders
		print(f"plotting \"{p.title}\"")

		nb_stacks = 0
		fix_dataset = []
		name_idx = {}
		i = 0
		for d in p.backends:
			y_stack = p.backends[d]["y_stack"]
			x_stack = p.backends[d]["x_stack"]
			if d.name in p.opt["filter_out_backends"]:
				continue
			l = len(y_stack)
			if l > nb_stacks:
				nb_stacks = l
			if l > 0:
				fix_dataset += [d]

		width = 0.83 / len(fix_dataset)
		offset = 0.005

		# print(json.dumps(stacked_bar_idx, indent=2))

		# fig, axs = plt.subplots(figsize=(f[0]*nb_stacks, f[1]), nrows=1, ncols=nb_stacks)
		# i = 0
		# for d in fix_dataset:
		first = fix_dataset[0] # TODO: some problem with the organization
		idx = 0

		
		plt.rcParams['hatch.linewidth'] = 0.1

		break_y_scale = p.opt["break_y_scale"]
		is_percent = p.opt["is_percent"]
		top_extra = p.opt["top_extra"]
		fix_100 = p.opt["fix_at_100_percent"]
		overide_max_y = p.opt["overide_max_y"]

		plt.tight_layout()
		offset_left_margin = 0
		# break_y = overide_max_y if not overide_max_y is None else max_y[s_title]
		# if break_y > 999:
		# 	offset_left_margin = 0.01
		# if break_y > 9999:
		# 	offset_left_margin = 0.026
		# if break_y > 99999:
		# 	offset_left_margin = 0.045
		# if break_y > 999999:
		# 	offset_left_margin = 0.05
		if fix_100:
			if top_extra < 0.5:
				plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.17+offset_left_margin, right=0.75, top=0.999, bottom=0.128)
			else:
				plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.19+offset_left_margin, right=0.998, top=0.99, bottom=0.128)
		else:
			plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.16+offset_left_margin, right=0.998, top=0.999, bottom=0.128)
			# plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.115, right=0.999, top=0.999, bottom=0.40)
			# print(s_title)
			# axs.set_title(f"{self.title}\n{s_title[0]}")
			axs.margins(x=0)
			
			axs.set_ylabel(p.y_label, size=16)
			axs.tick_params(axis='y', which='major', labelsize=13)
			if not axs_all is None:
				axs_all[1].tick_params(axis='y', which='major', labelsize=13)

			if not axs_all is None:
				axs.yaxis.set_label_coords(-.1-offset_left_margin, -.09)
			else:
				axs.yaxis.set_label_coords(-.11-offset_left_margin, .43)

			annot_loc = []

			# breakpoint()
			# (_, ss) = d.y_stack.items()
			#############
			for i,d in enumerate(p.backends):
				bottom = np.array([0 for _ in x_stack.transpose()])
				y_stack = p.backends[d]["y_stack"]
				x_stack = p.backends[d]["x_stack"]
				for sn, sy in y_stack.items(): #ss.items():
					# print(d.name, sn, "is_percent", is_percent)
					# j = plots_idx[s_title[0]]

					try:
						if is_percent:
							if not axs_all is None:
								for a in axs_all:
									a.yaxis.set_major_formatter(mtick.PercentFormatter())
							else:
								axs.yaxis.set_major_formatter(mtick.PercentFormatter())
							triple = [(np.average(x),np.average(y*100),np.std(y*100)) for x,y in zip(p.x_params[d.name].transpose(), sy.transpose())]
						else:
							triple = [(np.average(x),np.average(y),np.std(y)) for x,y in zip(p.x_params[d.name].transpose(), sy.transpose())]
						tripleF = triple
						if not p.opt["filter_x_fn"] is None:
							tripleF = list(filter(p.opt["filter_x_fn"], triple))
						tripleF.sort(key=lambda elem : elem[0]) # sort by X
						x_array, y_array, y_error = zip(*tripleF)
					except:
						# print(sn, f"idx={i} backend={d.name} {sy}")
						# breakpoint()
						continue
					xs = np.array([k for k in range(len(x_array))]) + i*width + i*offset
					ys = np.array(y_array)
					y_err = np.array(y_error)
					if len(bottom) > len(xs):
						bottom = np.array([0 for _ in xs])
					if not axs_all is None:
						for a in axs_all:
							a.bar(xs, ys, width, yerr = y_err, label=sn, bottom=bottom, color=p.colors[sn], hatch=p.patterns[sn])
					else:
						axs.bar(xs, ys, width, yerr = y_err, label=sn, bottom=bottom, color=p.colors[sn], hatch=p.patterns[sn])
					if i == 0: # print label
						if not axs_all is None:
							axs_all[0].legend(prop={'size': 14})
						else:
							if fix_100:
								if top_extra < 0.5:
									axs.legend(prop={'size': 13}, bbox_to_anchor=(0.99, 1.02))
								else:
									axs.legend(prop={'size': 13}, bbox_to_anchor=(1.01, 1.01), ncol=2)
							else:
								axs.legend(prop={'size': 14})
					bottom = bottom + ys
				for x,y in zip(xs,bottom):
					annot_loc += [(d.name, x,y)]
					break
				if not axs_all is None:
					axs_all[1].set_xlabel(p.x_label, size=14)
					for a in axs_all:
						a.set_xticks(np.array([k for k in range(len(x_array))]))
						a.set_xticklabels([f"    {int(x)}" for x in x_array], size=16)
						a.tick_params(axis='x', length=0, direction='in', width=0)
				else:
					axs.set_xlabel(p.x_label, size=14)
					axs.set_xticks(np.array([k for k in range(len(x_array))]))
					axs.set_xticklabels([f"    {int(x)}" for x in x_array], size=16)
					axs.tick_params(axis='x', length=0, direction='in', width=0)
			##############
			# breakpoint()
			annot_max_y = max([y for _,_,y in annot_loc ])
			for name, x,y in annot_loc:
				if not axs_all is None:
					axs_all[1].annotate(name, (x, annot_max_y), textcoords="offset points", xytext=(-6,4), ha='left', va='baseline', size=14, rotation=20)
					# axs_all[1].annotate(name, (x, 0), textcoords="offset points", xytext=(0,-4), ha='center', va='top', size=14, rotation=90)
				else:
					axs.annotate(name, (x, annot_max_y), textcoords="offset points", xytext=(-6,4), ha='left', va='baseline', size=14, rotation=20)

			bottom, top = axs.get_ylim()
			if is_percent and fix_100:
				top = 100
				axs.set_yticks([0, 20, 40, 60, 80, 100])
			else:
				axs.tick_params(axis='y', which='major', rotation=20, pad=-2)
			# print("top", top, "top_extra", top_extra)
			if axs_all is None:
				axs.set_ylim(top=top+top*top_extra, bottom=0)

			# plt.tight_layout()
			# print(f"stack_{idx}_{self.filename}")
			plt.savefig(f"{self.filename}")
			fig.clear()
			plt.close()
			idx += 1
