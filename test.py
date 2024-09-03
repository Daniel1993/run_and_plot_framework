#!/bin/env python3

from myplotlib.common import BenchmarkParameters, CollectData
from myplotlib.parse_sol import Parser
from myplotlib.plot import LinesPlot, StackPlot, BackendDataset
from myplotlib.plot import LineOptions, StackOptions
from os.path import exists

if __name__ == "__main__":
	params = BenchmarkParameters(["-w", "-m", "-s", "-d", "-o", "-p", "-r", "-n", "-t"])

	params.set_params("-s", [0], True) # set non-comb to avoid combinations all-to-all
	params.set_params("-o", [95], True)
	params.set_params("-p", [5], True)
	params.set_params("-r", [0], True)
	params.set_params("-d", [0], True)

	data_folder = "data-tpcc-motivation" # folder where the stdout will be saved

	params.set_params("-w", [64]) # nb warehouses
	params.set_params("-m", [64]) # max nb warehouses (put the same as -w)
	params.set_params("-t", [5])

	params.set_params("-n", [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32])
	nb_samples = 1

	locations = [
	 "../POWER8TM/benchmarks/tpcc", # location of the benchmark (TODO: use absolute paths)
	 "../POWER8TM/benchmarks/tpcc",
	]
	# The backend name goes here (don't forget to match the position in the
	# "backends" list with the position in the "locations" list)
	backends = [
	 "spht",
	 "spht-quiescence-naive2",
	]
# Label names in the plots
	name_map = {
		"psi" : "DUMBO-SI",
		"psi-strong" : "DUMBO-opaq",
		"psi-bug" : "DUMBO-SI-bug",
		"psi-strong-bug" : "DUMBO--opaq-bug",
		"spht-dumbo-readers" : "DUMBO-read",
		"spht" : "SPHT",
		"pstm" : "PSTM", 
		"spht-log-linking" : "SPHT-LL",
		"pisces" : "Pisces",
		"htm-sgl" : "HTM",
		"htm-sgl-sr" : "HTM+sus",
		"si-htm" : "SI-HTM",
		"ureads-strong": "ureads-strong", 
		"ureads-p8tm": "ureads-p8tm",
		"spht-quiescence-naive2": "SPHT+SIHTM",
	}
	
	l_o = LineOptions(
		"Throughput",                 # title
		"No. Threads",                # xlabel
		"Throughput (x1,000,000 T/s)" # ylabel
	).set_xaxis(
		lambda e: e["-n"] # var "e" is a dataframe with all the data from the .csv
	).set_yaxis(
		lambda e: e["txs-tpcc"]/e["time-tpcc"]/1000000
	).set_margins_offset( # from here below is plot adjustments (TODO: more to come)
		bottom=0.05
	).adjust_legend(
		size=20
	).adjust_title(
		size="x-large",
		pad=10
	).adjust_labels(
		pad=2
	).adjust_xticks(
		size=8
	)

	#####################################
	### Start aux functions for bar plots
	def filter_threads(t) -> bool:
		x, y, sd = t
		# return True on the threads to keep
		return True if x in [2, 4, 8, 16, 32] else False
	def divByAborts(e, attr):
		# if (e["total-aborts"] == 0).any():
		#   return 0
		# else:
		return (e[attr]/(e["total-aborts"]+e["total-commits"]-e["gl-commits"]))
	def checkIfUpdCommitStats(e, attr):
		if (e["total-upd-tx-time"] == 0).any():
			return [0 for i in range(len(e))]
		else:
			return (e[attr]/e["total-upd-tx-time"])
	def checkIfAbortStats(e, attr):
		if (e["total-abort-upd-tx-time"] == 0).any():
			return [0 for i in range(len(e))]
		if (e["total-upd-tx-time"] == 0).any():
			return [0 for i in range(len(e))]
		else:
			return (e[attr]/e["total-upd-tx-time"])
	def checkIfROCommitStats(e, attr):
		if (e["total-ro-tx-time"] == 0).any():
			return [0 for i in range(len(e))]
		else:
			return (e[attr]/e["total-ro-tx-time"])
	def checkIfROAbortStats(e, attr):
		if (e["total-abort-ro-tx-time"] == 0).any():
			return [0 for i in range(len(e))]
		if (e["total-ro-tx-time"] == 0).any():
			return [0 for i in range(len(e))]
		else:
			return (e[attr]/e["total-ro-tx-time"])
	### End aux functions for bar plots
	###################################

	diff_outcomes = StackOptions(
		"Prob. of different outcomes for a transaction",  # title
		"No. Threads",                                    # xlabel
		"Percentage of started transactions",             # ylabel
		StackOptions.y_FnBuilder().b(                     # y_fns
			"non-tx commit", lambda e: (e["nontx-commits"])/(e["total-commits"]+e["total-aborts"])
		).b(
			"non-tx commit", lambda e: (e["nontx-commits"])/(e["total-commits"]+e["total-aborts"]),
		).b(
			"ROT commit", lambda e: (e["rot-commits"])/(e["total-commits"]+e["total-aborts"]),
		).b(
			"HTM commit", lambda e: (e["htm-commits"])/(e["total-commits"]+e["total-aborts"]),
		).b(
			"SGL commit", lambda e: (e["gl-commits"])/(e["total-commits"]+e["total-aborts"]),
		).b(
			"STM commit", lambda e: (e["stm-commits"])/(e["total-commits"]+e["total-aborts"]),
		).b(
			"Abort", lambda e: (e["total-aborts"])/(e["total-commits"]+e["total-aborts"]),
		),
		is_percent = True,                                # other options (TODO: this will change)
		fix_at_100_percent = True,
		top_extra = 0.6,
		filter_x_fn = filter_threads
	).set_xaxis(
		lambda e: e["-n"]
	)
	aborts_types = StackOptions(
		"Abort types",                # title
		"No. Threads",                # xlabel
		"Percentage of aborts",       # ylabel
		StackOptions.y_FnBuilder().b( # y_fns
			"tx conflict", lambda e: (divByAborts(e, "confl-trans") + divByAborts(e, "rot-trans-aborts")),
		).b(
			"non-tx conflict", lambda e: (divByAborts(e, "confl-non-trans") + divByAborts(e, "rot-non-trans-aborts")),
		).b(
			"capacity", lambda e: (divByAborts(e, "capac-aborts") + divByAborts(e, "rot-capac-aborts")), 
		).b(
			"other", lambda e: (divByAborts(e, "other-aborts") + divByAborts(e, "confl-self") + divByAborts(e, "rot-self-aborts") + divByAborts(e, "user-aborts") + divByAborts(e, "rot-user-aborts")), 
		),
		is_percent=True,
		fix_at_100_percent=True,
		filter_x_fn=filter_threads
	).set_xaxis(
		lambda e: e["-n"]
	)
	lat_prof = StackOptions(
		"Latency profile (update txs)",       # title
		"No. Threads",                        # xlabel
		"Overhead over time processing txs",  # ylabel
		StackOptions.y_FnBuilder().b(         # y_fns
			#).b("processing committed txs.", lambda e: checkIfUpdCommitStats(e, "total-upd-tx-time")),
			"isolation wait", lambda e: (checkIfUpdCommitStats(e, "total-sus-time")),
		).b(
			#).b("suspend/resume", lambda e: (checkIfUpdCommitStats(e, "total-sus-time")),
			"redo log flush", lambda e: (checkIfUpdCommitStats(e, "total-flush-time")),
		).b(
			"durability wait", lambda e: (checkIfUpdCommitStats(e, "total-dur-commit-time")),
			#).b("proc. aborted txs", lambda e: (checkIfAbortStats(e, "total-abort-upd-tx-time")),
		),
		is_percent=True,
		top_extra=0.27,
		filter_x_fn=filter_threads
	).set_xaxis(
		lambda e: e["-n"]
	)
	lat_prof_RO = StackOptions(
		"Latency profile (read-only txs)",    # title
		"No. Threads",                        # xlabel
		"Overhead over time processing txs",  # ylabel
		StackOptions.y_FnBuilder().b(         # y_fns
			#).b("proc. committed txs", lambda e: checkIfROCommitStats(e, "total-ro-tx-time")),
			"durability\nwait", lambda e: (checkIfROCommitStats(e, "total-ro-dur-wait-time")),
			#).b("proc. aborted txs", lambda e: (checkIfROAbortStats(e, "total-abort-ro-tx-time"))
		),
		is_percent=True,
		filter_x_fn=filter_threads
	).set_xaxis(
		lambda e: e["-n"]
	)

	datasets_thr = {}
	datasets_aborts = {}
	for loc,backend in zip(locations,backends):
		for sample in range(nb_samples):
			data = CollectData(
					loc,
					"code/tpcc", # executable to run (TODO: absolute path is better)
					"build-tpcc.sh", # script to build
					backend,
					f"{data_folder}/{backend}-s{sample}"
				)
			# data.run_sample(params) # TODO: uncomment to run samples
			if not exists(f"{data_folder}/{backend}-s{sample}.csv"):
				parser = Parser(f"{data_folder}/{backend}-s{sample}")
				parser.parse_all(f"{data_folder}/{backend}-s{sample}.csv")
		lst_each = params.list_for_each_param(["-s", "-d", "-o", "-p", "-r"])

		for s,d,o,p,r in lst_each:
			ds = BackendDataset(
				name_map[backend],
				[f"{data_folder}/{backend}-s{sample}.csv" for sample in range(nb_samples)],
				{"-s": s, "-d": d, "-o": o, "-p": p, "-r": r}
			)
			ds.add_line(l_o)

			ds.add_stack(diff_outcomes)
		
			# Adds a bar plot for the abort type (TODO: not working) 
			ds.add_stack(aborts_types)      

			# Adds a bar plot for the profile information.
			ds.add_stack(lat_prof)

			# Adds a bar plot for the profile information.
			ds.add_stack(lat_prof_RO)

	aborts_types.add_labeled_color("SGL commit", "#a83232")
	aborts_types.add_labeled_color("Abort", "#D0D0D0")
		
	l_o.plot()
	diff_outcomes.plot()
	aborts_types.plot()
	lat_prof.plot()
	lat_prof_RO.plot()
