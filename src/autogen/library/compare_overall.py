from . import compare_test as cpre
from . import compare_utils
from . import print_terms as pt
#arguments: list of terms, face factor, last means whether this is the last commutator or not (0,1)
def compare_envelope(list_terms, fc,last):
    #compare terms based on 5 levels of check all in cpre.compare()
    #for i in range(1):
        #for j in range(1,2):

    def merge_terms(rep, term, flo):
        #print "comparing:", rep.fac, rep.coeff_list, term.fac, term.coeff_list, flo
        #print 'in result in the comparision', i, j, flo
        #print 'this should be 0 always = ', rep.fac + term.fac * flo
        #Unless the sign of the terms is not being calculated
        if rep.fac < 0.0:
            rep.fac = rep.fac - abs(term.fac)
        else:
            rep.fac = rep.fac + abs(term.fac)
        term.fac = 0.0
        #print 'result in compare when matched', rep.fac, term.fac, flo

    compare_utils.reduce_terms_two_stage(
        list_terms,
        cpre.compare,
        merge_terms,
        key_func=compare_utils.coarse_key,
        secondary_key_func=compare_utils.matrix_key,
    )

    #muliply with the prefactor of the expression from the Housdoff Expression
    #for item in list_terms:
        #if item.fac!=0.0:
            #item.fac=item.fac*fc
    
    #print terms properly
    if last!=0:
            
        #pt.print_terms(list_terms)
        #delete operator coefficient in self.coeff_list
        

        for term in list_terms:

            if len(term.coeff_list)==len(term.map_org)+1:
                #print 'deleting operator coeff'
                term.coeff_list.pop()
            elif len(term.coeff_list)>len(term.map_org):
                print(' in compare envolope terminal error')
                sys.exit(0)
    ##list_terms=pt.clean_list(list_terms)
    return list_terms
