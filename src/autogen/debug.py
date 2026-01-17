from __future__ import annotations

from autogen.library import convert_pqr
from autogen.library import make_op
from autogen.library import print_terms
from autogen.library import class_term
from autogen.pkg import func_ewt


def general_term(ops):
	"""Create a single combined term for debug output.

	This used to live in tests, but debug is part of the installable package.
	"""
	dict_ind = {}
	list_op, dict_ind = make_op.make_op(ops, dict_ind)
	sum1 = []
	coeff = []
	op = func_ewt.contractedobj('op', 1, 1)
	st = [[op]]
	co = []
	term = class_term.term(1, sum1, coeff, list_op, st, co)
	term.dict_ind = dict_ind
	term.map_org = list_op
	for item in list_op:
		term.coeff_list.append(item.coeff)
		term.sum_list.extend(item.sum_ind)
		term.st[0][-1].upper.extend(item.st[0][-1].upper)
		term.st[0][-1].lower.extend(item.st[0][-1].lower)
	print(term.st[0][-1].upper)
	print(term.map_org, term.dict_ind)
	return term


def run_debug(output_file: str = "latex_output.txt"):
	term = general_term(["V2", "T2", "T1"])
	list_terms = [term]

	print_terms.print_terms(list_terms, output_file)
	list_terms = convert_pqr.convert_pqr(list_terms)
	print_terms.print_terms(list_terms, output_file)
	return list_terms


if __name__ == "__main__":
	run_debug()
