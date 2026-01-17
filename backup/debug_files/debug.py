from autogen.main_tools import product as math
from autogen.library import convert_pqr
from autogen.main_tools.commutator import comm
from autogen.library import pick
from autogen.library import print_terms
from autogen.library import rn_comm
from autogen.debug import general_term
#comm(['V2'],['T2'],1)
#list_terms=comm(comm(['V2'],['T2'],0),['T1'],1)
#list_terms=comm(['V2'],['T2'],1)
#print_terms.print_terms(list_terms)
#'''
term=general_term(['V2','T2','T1'])
list_terms=[term]

print_terms.print_terms(list_terms)
print('general term creation done')
print('printing done')
list_terms=convert_pqr.convert_pqr(list_terms)

#list_terms=print_terms.clean_list(list_terms)
#list_terms=pick.pick(list_terms, ['i'],['j'])

print('------')
print_terms.print_terms(list_terms)
print('------')
print('conversion done')
#for item in list_terms:
#    print rn_comm.rank(item)

#list_terms=rn_comm.select_r(list_terms)

#print_terms.print_terms(list_terms)
#'''
