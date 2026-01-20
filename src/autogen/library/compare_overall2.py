from . import compare_test_outside as cpre
from . import compare_utils
from . import print_terms as pt
#arguments: list of terms, face factor, last means whether this is the last commutator or not (0,1)
def compare_envelope(list_terms, fc,last):
    #compare terms based on 5 levels of check all in cpre.compare()
    #for i in range(1):
        #for j in range(1,2):


    for bucket in compare_utils.bucket_terms(list_terms):
        for idx_i, i in enumerate(bucket):
            if list_terms[i].fac == 0:
                continue
            for j in bucket[idx_i + 1 :]:
                if list_terms[j].fac == 0:
                    continue
                #print 'non zero compare'
                flo = cpre.compare(list_terms[i], list_terms[j])
                #if flo!=0:
                #    #print "comparing:", list_terms[i].fac,list_terms[i].coeff_list,list_terms[j].fac,list_terms[j].coeff_list, flo
                #    #print 'in result in the comparision',i,j,flo
                #    #why? (on below statement- 18Feb2020)
                #    #print 'this should be 0 always = ',list_terms[i].fac+list_terms[j].fac*flo
                #    #Unless the sign of the terms is not being calculated
                #    if (list_terms[i].fac<0.0): 
                #        list_terms[i].fac=list_terms[i].fac - abs(list_terms[j].fac)
                #    else:
                #        list_terms[i].fac=list_terms[i].fac + abs(list_terms[j].fac)
                #    list_terms[j].fac=0.0
                #    #print 'result in compare when matched',list_terms[i].fac,list_terms[j].fac,flo
    #list_terms=pt.clean_list(list_terms)
    #print 'length in compare',len(list_terms)
    #exit()

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


    return list_terms
