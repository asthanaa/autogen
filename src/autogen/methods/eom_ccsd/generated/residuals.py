import numpy as np

AUTOGEN_SPIN_SUMMED = True
AUTOGEN_SPIN_SUMMED_MODE = 'direct'
AUTOGEN_INTERMEDIATES = True
VIEW_TENSORS = ('f', 'g')
OCC = set('ijklmn')
VIRT = set('abcdefgh')

def view_tensor(tensor, labels, o, v):
    idx = []
    list_axes = []
    for axis, label in enumerate(labels):
        if label and label[0] in OCC:
            idx.append(o)
            list_axes.append(axis)
        elif label and label[0] in VIRT:
            idx.append(v)
            list_axes.append(axis)
        else:
            idx.append(slice(None))
    if not list_axes:
        return tensor[tuple(idx)]
    ix = np.ix_(*[idx[a] for a in list_axes])
    ix_iter = iter(ix)
    full_idx = []
    for axis in range(len(idx)):
        if axis in list_axes:
            full_idx.append(next(ix_iter))
        else:
            full_idx.append(idx[axis])
    return tensor[tuple(full_idx)]

def zeros_for_output(labels, o, v):
    if not labels:
        return 0.0
    shape = []
    for label in labels:
        if label and label[0] in OCC:
            shape.append(len(o))
        elif label and label[0] in VIRT:
            shape.append(len(v))
        else:
            shape.append(len(o) + len(v))
    return np.zeros(tuple(shape))

def _get_viewer(tensor_map, o, v):
    views = {}
    def get_view(name, labels):
        key = (name, labels)
        if key in views:
            return views[key]
        tensor = tensor_map[name]
        views[key] = view_tensor(tensor, labels, o, v)
        return views[key]
    return get_view

def compute_intermediates(get_view, t1, t2, r1, r2):
    I0 = np.einsum('klcd,bk->bcdl', get_view('g', 'klcd'), t1, optimize=True)
    I1 = np.einsum('klcd,ak->acdl', get_view('g', 'klcd'), t1, optimize=True)
    I2 = np.einsum('klcd,bl->bcdk', get_view('g', 'klcd'), r1, optimize=True)
    I3 = np.einsum('klcd,al->acdk', get_view('g', 'klcd'), r1, optimize=True)
    I4 = np.einsum('klcd,cj->djkl', get_view('g', 'klcd'), t1, optimize=True)
    I5 = np.einsum('klcd,dj->cjkl', get_view('g', 'klcd'), t1, optimize=True)
    I6 = np.einsum('klcd,di->cikl', get_view('g', 'klcd'), t1, optimize=True)
    I7 = np.einsum('klcd,ci->dikl', get_view('g', 'klcd'), t1, optimize=True)
    I8 = np.einsum('klcd,abkj->abcdjl', get_view('g', 'klcd'), t2, optimize=True)
    I9 = np.einsum('klcd,abik->abcdil', get_view('g', 'klcd'), t2, optimize=True)
    I10 = np.einsum('jkbc,aj->abck', get_view('g', 'jkbc'), t1, optimize=True)
    I11 = np.einsum('klcd,cj->djkl', get_view('g', 'klcd'), r1, optimize=True)
    I12 = np.einsum('klcd,di->cikl', get_view('g', 'klcd'), r1, optimize=True)
    I13 = np.einsum('klcd,dj->cjkl', get_view('g', 'klcd'), r1, optimize=True)
    I14 = np.einsum('jkbc,ak->abcj', get_view('g', 'jkbc'), r1, optimize=True)
    I15 = np.einsum('klcd,al->acdk', get_view('g', 'klcd'), t1, optimize=True)
    I17 = np.einsum('klcd,dk->cl', get_view('g', 'klcd'), t1, optimize=True)
    I20 = np.einsum('klcd,ci->dikl', get_view('g', 'klcd'), r1, optimize=True)
    I21 = np.einsum('klcd,dbij->bcijkl', get_view('g', 'klcd'), t2, optimize=True)
    I22 = np.einsum('klcd,cbij->bdijkl', get_view('g', 'klcd'), t2, optimize=True)
    I23 = np.einsum('klcd,abil->abcdik', get_view('g', 'klcd'), r2, optimize=True)
    I24 = np.einsum('klcd,ablj->abcdjk', get_view('g', 'klcd'), r2, optimize=True)
    I25 = np.einsum('klcd,acij->adijkl', get_view('g', 'klcd'), t2, optimize=True)
    I26 = np.einsum('klcd,adij->acijkl', get_view('g', 'klcd'), t2, optimize=True)
    I27 = np.einsum('klcd,ablk->abcd', get_view('g', 'klcd'), t2, optimize=True)
    I28 = np.einsum('klcd,abkl->abcd', get_view('g', 'klcd'), t2, optimize=True)
    I29 = np.einsum('klcd,dbkj->bcjl', get_view('g', 'klcd'), t2, optimize=True)
    I30 = np.einsum('klcd,cbkj->bdjl', get_view('g', 'klcd'), t2, optimize=True)
    I31 = np.einsum('klcd,cbjk->bdjl', get_view('g', 'klcd'), t2, optimize=True)
    I32 = np.einsum('klcd,dcij->ijkl', get_view('g', 'klcd'), t2, optimize=True)
    I33 = np.einsum('klcd,cdij->ijkl', get_view('g', 'klcd'), t2, optimize=True)
    I34 = np.einsum('klcd,caik->adil', get_view('g', 'klcd'), t2, optimize=True)
    I35 = np.einsum('klcd,caki->adil', get_view('g', 'klcd'), t2, optimize=True)
    I36 = np.einsum('klcd,acik->adil', get_view('g', 'klcd'), t2, optimize=True)
    I37 = np.einsum('klcd,adik->acil', get_view('g', 'klcd'), t2, optimize=True)
    I38 = np.einsum('kbcd,dj->bcjk', get_view('g', 'kbcd'), t1, optimize=True)
    I39 = np.einsum('jkbc,ci->bijk', get_view('g', 'jkbc'), t1, optimize=True)
    I40 = np.einsum('jkbc,bi->cijk', get_view('g', 'jkbc'), t1, optimize=True)
    I42 = np.einsum('kljc,bk->bcjl', get_view('g', 'kljc'), t1, optimize=True)
    I44 = np.einsum('klcd,adkj->acjl', get_view('g', 'klcd'), t2, optimize=True)
    I45 = np.einsum('klcd,dbik->bcil', get_view('g', 'klcd'), t2, optimize=True)
    I46 = np.einsum('klcd,cblj->bdjk', get_view('g', 'klcd'), r2, optimize=True)
    I49 = np.einsum('klcd,cbij->bdijkl', get_view('g', 'klcd'), r2, optimize=True)
    I50 = np.einsum('klcd,dbij->bcijkl', get_view('g', 'klcd'), r2, optimize=True)
    I51 = np.einsum('kbcd,ak->abcd', get_view('g', 'kbcd'), t1, optimize=True)
    I52 = np.einsum('kacd,bk->abcd', get_view('g', 'kacd'), t1, optimize=True)
    I53 = np.einsum('kacd,di->acik', get_view('g', 'kacd'), t1, optimize=True)
    I54 = np.einsum('kbcd,dj->bcjk', get_view('g', 'kbcd'), r1, optimize=True)
    I55 = np.einsum('jkbc,cj->bk', get_view('g', 'jkbc'), t1, optimize=True)
    I56 = np.einsum('jkbc,bj->ck', get_view('g', 'jkbc'), t1, optimize=True)
    I57 = np.einsum('jkbc,bk->cj', get_view('g', 'jkbc'), r1, optimize=True)
    I58 = np.einsum('jkbc,ck->bj', get_view('g', 'jkbc'), r1, optimize=True)
    I59 = np.einsum('jkbc,bi->cijk', get_view('g', 'jkbc'), r1, optimize=True)
    I60 = np.einsum('jkbc,ci->bijk', get_view('g', 'jkbc'), r1, optimize=True)
    I63 = np.einsum('klcj,bl->bcjk', get_view('g', 'klcj'), r1, optimize=True)
    I67 = np.einsum('klcd,acil->adik', get_view('g', 'klcd'), r2, optimize=True)
    I70 = np.einsum('klcd,acij->adijkl', get_view('g', 'klcd'), r2, optimize=True)
    I71 = np.einsum('klcd,adij->acijkl', get_view('g', 'klcd'), r2, optimize=True)
    I72 = np.einsum('kacd,cj->adjk', get_view('g', 'kacd'), t1, optimize=True)
    I73 = np.einsum('kbcd,ci->bdik', get_view('g', 'kbcd'), t1, optimize=True)
    I74 = np.einsum('kacd,di->acik', get_view('g', 'kacd'), r1, optimize=True)
    I75 = np.einsum('kbcd,ak->abcd', get_view('g', 'kbcd'), r1, optimize=True)
    I76 = np.einsum('kacd,bk->abcd', get_view('g', 'kacd'), r1, optimize=True)
    I77 = np.einsum('klci,bk->bcil', get_view('g', 'klci'), t1, optimize=True)
    I78 = np.einsum('klic,ak->acil', get_view('g', 'klic'), t1, optimize=True)
    I79 = np.einsum('klci,al->acik', get_view('g', 'klci'), r1, optimize=True)
    return {
        'I0': I0,
        'I1': I1,
        'I2': I2,
        'I3': I3,
        'I4': I4,
        'I5': I5,
        'I6': I6,
        'I7': I7,
        'I8': I8,
        'I9': I9,
        'I10': I10,
        'I11': I11,
        'I12': I12,
        'I13': I13,
        'I14': I14,
        'I15': I15,
        'I17': I17,
        'I20': I20,
        'I21': I21,
        'I22': I22,
        'I23': I23,
        'I24': I24,
        'I25': I25,
        'I26': I26,
        'I27': I27,
        'I28': I28,
        'I29': I29,
        'I30': I30,
        'I31': I31,
        'I32': I32,
        'I33': I33,
        'I34': I34,
        'I35': I35,
        'I36': I36,
        'I37': I37,
        'I38': I38,
        'I39': I39,
        'I40': I40,
        'I42': I42,
        'I44': I44,
        'I45': I45,
        'I46': I46,
        'I49': I49,
        'I50': I50,
        'I51': I51,
        'I52': I52,
        'I53': I53,
        'I54': I54,
        'I55': I55,
        'I56': I56,
        'I57': I57,
        'I58': I58,
        'I59': I59,
        'I60': I60,
        'I63': I63,
        'I67': I67,
        'I70': I70,
        'I71': I71,
        'I72': I72,
        'I73': I73,
        'I74': I74,
        'I75': I75,
        'I76': I76,
        'I77': I77,
        'I78': I78,
        'I79': I79,
    }

def compute_outputs(f, g, t1, t2, r1, r2, o, v):
    tensor_map = {
        'f': f,
        'g': g,
    }
    get_view = _get_viewer(tensor_map, o, v)
    inter = compute_intermediates(get_view, t1, t2, r1, r2)
    I0 = inter['I0']
    I1 = inter['I1']
    I2 = inter['I2']
    I3 = inter['I3']
    I4 = inter['I4']
    I5 = inter['I5']
    I6 = inter['I6']
    I7 = inter['I7']
    I8 = inter['I8']
    I9 = inter['I9']
    I10 = inter['I10']
    I11 = inter['I11']
    I12 = inter['I12']
    I13 = inter['I13']
    I14 = inter['I14']
    I15 = inter['I15']
    I17 = inter['I17']
    I20 = inter['I20']
    I21 = inter['I21']
    I22 = inter['I22']
    I23 = inter['I23']
    I24 = inter['I24']
    I25 = inter['I25']
    I26 = inter['I26']
    I27 = inter['I27']
    I28 = inter['I28']
    I29 = inter['I29']
    I30 = inter['I30']
    I31 = inter['I31']
    I32 = inter['I32']
    I33 = inter['I33']
    I34 = inter['I34']
    I35 = inter['I35']
    I36 = inter['I36']
    I37 = inter['I37']
    I38 = inter['I38']
    I39 = inter['I39']
    I40 = inter['I40']
    I42 = inter['I42']
    I44 = inter['I44']
    I45 = inter['I45']
    I46 = inter['I46']
    I49 = inter['I49']
    I50 = inter['I50']
    I51 = inter['I51']
    I52 = inter['I52']
    I53 = inter['I53']
    I54 = inter['I54']
    I55 = inter['I55']
    I56 = inter['I56']
    I57 = inter['I57']
    I58 = inter['I58']
    I59 = inter['I59']
    I60 = inter['I60']
    I63 = inter['I63']
    I67 = inter['I67']
    I70 = inter['I70']
    I71 = inter['I71']
    I72 = inter['I72']
    I73 = inter['I73']
    I74 = inter['I74']
    I75 = inter['I75']
    I76 = inter['I76']
    I77 = inter['I77']
    I78 = inter['I78']
    I79 = inter['I79']
    outputs = {}
    s1 = zeros_for_output('ai', o, v)
    s1 += (0.9999999999999998) * np.einsum('ab,bi->ai', get_view('f', 'ab'), r1, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('abij,bj->ai', t2, I58, optimize=True)
    s1 += (-0.4999999999999999) * np.einsum('abik,bk->ai', r2, I55, optimize=True)
    s1 += (-0.24999999999999994) * np.einsum('abkj,bijk->ai', r2, I39, optimize=True)
    s1 += (-0.24999999999999994) * np.einsum('abkj,bijk->ai', t2, I60, optimize=True)
    s1 += (-0.4999999999999999) * np.einsum('acij,cj->ai', t2, I57, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('acik,ck->ai', r2, I56, optimize=True)
    s1 += (-0.24999999999999994) * np.einsum('acjk,cijk->ai', r2, I40, optimize=True)
    s1 += (-0.24999999999999994) * np.einsum('acjk,cijk->ai', t2, I59, optimize=True)
    s1 += (-1.9999999999999996) * np.einsum('baij,bj->ai', t2, I58, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('baik,bk->ai', r2, I55, optimize=True)
    s1 += (2.9999999999999996) * np.einsum('baji,bj->ai', t2, I58, optimize=True)
    s1 += (-0.7499999999999998) * np.einsum('bajk,bijk->ai', r2, I39, optimize=True)
    s1 += (-0.7499999999999998) * np.einsum('bajk,bijk->ai', t2, I60, optimize=True)
    s1 += (-1.4999999999999996) * np.einsum('baki,bk->ai', r2, I55, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('bakj,bijk->ai', r2, I39, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('bakj,bijk->ai', t2, I60, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('bcij,abcj->ai', t2, I14, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('bcik,abck->ai', r2, I10, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('bcji,abcj->ai', t2, I14, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('bcki,abck->ai', r2, I10, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('bi,cj,abcj->ai', t1, t1, I14, optimize=True)
    s1 += (-1.9999999999999996) * np.einsum('bi,ck,abck->ai', t1, r1, I10, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('bk,ci,abck->ai', t1, r1, I10, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('caij,cj->ai', t2, I57, optimize=True)
    s1 += (-1.9999999999999996) * np.einsum('caik,ck->ai', r2, I56, optimize=True)
    s1 += (-1.4999999999999996) * np.einsum('caji,cj->ai', t2, I57, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('cajk,cijk->ai', r2, I40, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('cajk,cijk->ai', t2, I59, optimize=True)
    s1 += (2.9999999999999996) * np.einsum('caki,ck->ai', r2, I56, optimize=True)
    s1 += (-0.7499999999999998) * np.einsum('cakj,cijk->ai', r2, I40, optimize=True)
    s1 += (-0.7499999999999998) * np.einsum('cakj,cijk->ai', t2, I59, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('cbij,abcj->ai', t2, I14, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('cbik,abck->ai', r2, I10, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('cbji,abcj->ai', t2, I14, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('cbki,abck->ai', r2, I10, optimize=True)
    s1 += (-1.9999999999999996) * np.einsum('ci,bj,abcj->ai', t1, t1, I14, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('ci,bk,abck->ai', t1, r1, I10, optimize=True)
    s1 += (-1.9999999999999996) * np.einsum('ck,bi,abck->ai', t1, r1, I10, optimize=True)
    s1 += (-0.4999999999999999) * np.einsum('jabc,bcij->ai', get_view('g', 'jabc'), r2, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('jabc,bcji->ai', get_view('g', 'jabc'), r2, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('jabc,bi,cj->ai', get_view('g', 'jabc'), t1, r1, optimize=True)
    s1 += (1.9999999999999996) * np.einsum('jabc,bj,ci->ai', get_view('g', 'jabc'), t1, r1, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('jabc,cbij->ai', get_view('g', 'jabc'), r2, optimize=True)
    s1 += (-0.4999999999999999) * np.einsum('jabc,cbji->ai', get_view('g', 'jabc'), r2, optimize=True)
    s1 += (1.9999999999999996) * np.einsum('jabc,ci,bj->ai', get_view('g', 'jabc'), t1, r1, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('jabc,cj,bi->ai', get_view('g', 'jabc'), t1, r1, optimize=True)
    s1 += (1.9999999999999996) * np.einsum('jabi,bj->ai', get_view('g', 'jabi'), r1, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('jaib,bj->ai', get_view('g', 'jaib'), r1, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('jb,abij->ai', get_view('f', 'jb'), r2, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('jb,aj,bi->ai', get_view('f', 'jb'), t1, r1, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('jb,baij->ai', get_view('f', 'jb'), r2, optimize=True)
    s1 += (1.4999999999999996) * np.einsum('jb,baji->ai', get_view('f', 'jb'), r2, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('jb,bi,aj->ai', get_view('f', 'jb'), t1, r1, optimize=True)
    s1 += (-0.9999999999999998) * np.einsum('ji,aj->ai', get_view('f', 'ji'), r1, optimize=True)
    s1 += (-0.24999999999999994) * np.einsum('jkbi,abkj->ai', get_view('g', 'jkbi'), r2, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('jkbi,aj,bk->ai', get_view('g', 'jkbi'), t1, r1, optimize=True)
    s1 += (-0.7499999999999998) * np.einsum('jkbi,bajk->ai', get_view('g', 'jkbi'), r2, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('jkbi,bakj->ai', get_view('g', 'jkbi'), r2, optimize=True)
    s1 += (-1.9999999999999996) * np.einsum('jkbi,bj,ak->ai', get_view('g', 'jkbi'), t1, r1, optimize=True)
    s1 += (-0.24999999999999994) * np.einsum('jkib,abjk->ai', get_view('g', 'jkib'), r2, optimize=True)
    s1 += (-1.9999999999999996) * np.einsum('jkib,aj,bk->ai', get_view('g', 'jkib'), t1, r1, optimize=True)
    s1 += (0.4999999999999999) * np.einsum('jkib,bajk->ai', get_view('g', 'jkib'), r2, optimize=True)
    s1 += (-0.7499999999999998) * np.einsum('jkib,bakj->ai', get_view('g', 'jkib'), r2, optimize=True)
    s1 += (0.9999999999999998) * np.einsum('jkib,bj,ak->ai', get_view('g', 'jkib'), t1, r1, optimize=True)
    outputs['s1'] = s1
    s2 = zeros_for_output('abij', o, v)
    s2 += (1.0) * np.einsum('abik,cl,cjkl->abij', t2, r1, I5, optimize=True)
    s2 += (-2.0) * np.einsum('abik,dl,djkl->abij', t2, r1, I4, optimize=True)
    s2 += (1.0) * np.einsum('abkj,cl,cikl->abij', t2, r1, I6, optimize=True)
    s2 += (-2.0) * np.einsum('abkj,dl,dikl->abij', t2, r1, I7, optimize=True)
    s2 += (0.5) * np.einsum('abkl,ci,cjkl->abij', t2, r1, I5, optimize=True)
    s2 += (0.5) * np.einsum('abkl,dj,dikl->abij', t2, r1, I7, optimize=True)
    s2 += (0.5) * np.einsum('abkl,ijkl->abij', r2, I33, optimize=True)
    s2 += (0.5) * np.einsum('ablj,ci,cl->abij', t2, r1, I17, optimize=True)
    s2 += (0.5) * np.einsum('ablk,cj,cikl->abij', t2, r1, I6, optimize=True)
    s2 += (0.5) * np.einsum('ablk,di,djkl->abij', t2, r1, I4, optimize=True)
    s2 += (0.5) * np.einsum('ablk,ijkl->abij', r2, I32, optimize=True)
    s2 += (1.0) * np.einsum('ac,cbij->abij', get_view('f', 'ac'), r2, optimize=True)
    s2 += (0.5) * np.einsum('acij,dk,bcdk->abij', t2, t1, I2, optimize=True)
    s2 += (-2.0) * np.einsum('acij,dl,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (1.0) * np.einsum('acik,bcjk->abij', r2, I38, optimize=True)
    s2 += (1.0) * np.einsum('acik,bcjk->abij', t2, I54, optimize=True)
    s2 += (-1.0) * np.einsum('acik,bcjk->abij', t2, I63, optimize=True)
    s2 += (-0.5) * np.einsum('acik,bl,cjkl->abij', t2, t1, I13, optimize=True)
    s2 += (-0.5) * np.einsum('acik,dj,bcdk->abij', t2, t1, I2, optimize=True)
    s2 += (-1.0) * np.einsum('acil,bcjl->abij', r2, I29, optimize=True)
    s2 += (-1.0) * np.einsum('acil,bcjl->abij', r2, I42, optimize=True)
    s2 += (0.5) * np.einsum('acil,dj,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (1.0) * np.einsum('aclj,bcil->abij', r2, I45, optimize=True)
    s2 += (1.0) * np.einsum('aclj,bcil->abij', r2, I77, optimize=True)
    s2 += (0.5) * np.einsum('aclj,di,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (-0.5) * np.einsum('aclk,bcijkl->abij', r2, I21, optimize=True)
    s2 += (-0.5) * np.einsum('aclk,bcijkl->abij', t2, I50, optimize=True)
    s2 += (-1.0) * np.einsum('adij,ck,bcdk->abij', t2, t1, I2, optimize=True)
    s2 += (1.0) * np.einsum('adij,cl,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (0.5) * np.einsum('adik,bl,djkl->abij', t2, t1, I11, optimize=True)
    s2 += (0.5) * np.einsum('adik,cj,bcdk->abij', t2, t1, I2, optimize=True)
    s2 += (2.0) * np.einsum('adil,bdjl->abij', r2, I30, optimize=True)
    s2 += (-1.0) * np.einsum('adil,bdjl->abij', r2, I31, optimize=True)
    s2 += (-0.5) * np.einsum('adil,cj,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (-1.0) * np.einsum('adkj,bdik->abij', r2, I73, optimize=True)
    s2 += (0.5) * np.einsum('adkj,bl,dikl->abij', t2, t1, I20, optimize=True)
    s2 += (0.5) * np.einsum('adkj,ci,bcdk->abij', t2, t1, I2, optimize=True)
    s2 += (-0.5) * np.einsum('adkl,bdijkl->abij', r2, I22, optimize=True)
    s2 += (-0.5) * np.einsum('adkl,bdijkl->abij', t2, I49, optimize=True)
    s2 += (-1.0) * np.einsum('ak,ci,bcjk->abij', t1, r1, I38, optimize=True)
    s2 += (1.0) * np.einsum('ak,ci,bcjk->abij', t1, t1, I63, optimize=True)
    s2 += (1.0) * np.einsum('al,ci,bcjl->abij', t1, r1, I42, optimize=True)
    s2 += (1.0) * np.einsum('al,cj,bcil->abij', t1, r1, I77, optimize=True)
    s2 += (1.0) * np.einsum('al,cj,di,bcdl->abij', t1, t1, r1, I0, optimize=True)
    s2 += (1.0) * np.einsum('al,dcij,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (1.0) * np.einsum('al,di,cj,bcdl->abij', t1, t1, r1, I0, optimize=True)
    s2 += (1.0) * np.einsum('bacd,cj,di->abij', get_view('g', 'bacd'), t1, r1, optimize=True)
    s2 += (1.0) * np.einsum('bacd,dcij->abij', get_view('g', 'bacd'), r2, optimize=True)
    s2 += (1.0) * np.einsum('bacd,di,cj->abij', get_view('g', 'bacd'), t1, r1, optimize=True)
    s2 += (1.0) * np.einsum('baci,cj->abij', get_view('g', 'baci'), r1, optimize=True)
    s2 += (1.0) * np.einsum('bajc,ci->abij', get_view('g', 'bajc'), r1, optimize=True)
    s2 += (1.0) * np.einsum('bc,acij->abij', get_view('f', 'bc'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('caik,bcjk->abij', r2, I38, optimize=True)
    s2 += (-1.0) * np.einsum('caik,bcjk->abij', t2, I54, optimize=True)
    s2 += (1.0) * np.einsum('caik,bcjk->abij', t2, I63, optimize=True)
    s2 += (0.5) * np.einsum('caik,bl,cjkl->abij', t2, t1, I13, optimize=True)
    s2 += (0.5) * np.einsum('caik,dj,bcdk->abij', t2, t1, I2, optimize=True)
    s2 += (1.0) * np.einsum('cail,bcjl->abij', r2, I29, optimize=True)
    s2 += (1.0) * np.einsum('cail,bcjl->abij', r2, I42, optimize=True)
    s2 += (1.0) * np.einsum('caki,bcjk->abij', r2, I38, optimize=True)
    s2 += (1.0) * np.einsum('caki,bcjk->abij', t2, I54, optimize=True)
    s2 += (-1.0) * np.einsum('caki,bcjk->abij', t2, I63, optimize=True)
    s2 += (-0.5) * np.einsum('caki,bl,cjkl->abij', t2, t1, I13, optimize=True)
    s2 += (-0.5) * np.einsum('caki,dj,bcdk->abij', t2, t1, I2, optimize=True)
    s2 += (-0.5) * np.einsum('cakl,bcijkl->abij', r2, I21, optimize=True)
    s2 += (-0.5) * np.einsum('cakl,bcijkl->abij', t2, I50, optimize=True)
    s2 += (-1.0) * np.einsum('cali,bcjl->abij', r2, I29, optimize=True)
    s2 += (-1.0) * np.einsum('cali,bcjl->abij', r2, I42, optimize=True)
    s2 += (0.5) * np.einsum('calk,bcijkl->abij', r2, I21, optimize=True)
    s2 += (0.5) * np.einsum('calk,bcijkl->abij', t2, I50, optimize=True)
    s2 += (0.5) * np.einsum('cbij,dk,acdk->abij', t2, t1, I3, optimize=True)
    s2 += (-2.0) * np.einsum('cbij,dl,acdl->abij', t2, r1, I1, optimize=True)
    s2 += (1.0) * np.einsum('cbil,acjl->abij', r2, I44, optimize=True)
    s2 += (0.5) * np.einsum('cbil,dj,acdl->abij', t2, r1, I1, optimize=True)
    s2 += (-1.0) * np.einsum('cbjk,acik->abij', r2, I53, optimize=True)
    s2 += (-1.0) * np.einsum('cbjk,acik->abij', t2, I74, optimize=True)
    s2 += (1.0) * np.einsum('cbjk,acik->abij', t2, I79, optimize=True)
    s2 += (0.5) * np.einsum('cbjk,al,cikl->abij', t2, t1, I12, optimize=True)
    s2 += (0.5) * np.einsum('cbjk,di,acdk->abij', t2, t1, I3, optimize=True)
    s2 += (1.0) * np.einsum('cbjl,acil->abij', r2, I37, optimize=True)
    s2 += (1.0) * np.einsum('cbjl,acil->abij', r2, I78, optimize=True)
    s2 += (2.0) * np.einsum('cbkj,acik->abij', r2, I53, optimize=True)
    s2 += (2.0) * np.einsum('cbkj,acik->abij', t2, I74, optimize=True)
    s2 += (-2.0) * np.einsum('cbkj,acik->abij', t2, I79, optimize=True)
    s2 += (-1.0) * np.einsum('cbkj,al,cikl->abij', t2, t1, I12, optimize=True)
    s2 += (-1.0) * np.einsum('cbkj,di,acdk->abij', t2, t1, I3, optimize=True)
    s2 += (-1.0) * np.einsum('cbkl,acijkl->abij', r2, I26, optimize=True)
    s2 += (-1.0) * np.einsum('cbkl,acijkl->abij', t2, I71, optimize=True)
    s2 += (-1.0) * np.einsum('cblj,acil->abij', r2, I37, optimize=True)
    s2 += (-2.0) * np.einsum('cblj,acil->abij', r2, I78, optimize=True)
    s2 += (0.5) * np.einsum('cblj,di,acdl->abij', t2, r1, I1, optimize=True)
    s2 += (0.5) * np.einsum('cblk,acijkl->abij', r2, I26, optimize=True)
    s2 += (0.5) * np.einsum('cblk,acijkl->abij', t2, I71, optimize=True)
    s2 += (0.5) * np.einsum('cdij,abcd->abij', r2, I28, optimize=True)
    s2 += (-1.0) * np.einsum('cdij,abcd->abij', r2, I51, optimize=True)
    s2 += (-1.0) * np.einsum('cdij,abcd->abij', t2, I75, optimize=True)
    s2 += (1.0) * np.einsum('cdij,bl,acdl->abij', t2, r1, I1, optimize=True)
    s2 += (0.5) * np.einsum('cdik,abcdjk->abij', t2, I24, optimize=True)
    s2 += (-1.5) * np.einsum('cdil,abcdjl->abij', r2, I8, optimize=True)
    s2 += (0.5) * np.einsum('cdjk,abcdik->abij', t2, I23, optimize=True)
    s2 += (-0.5) * np.einsum('cdjl,abcdil->abij', r2, I9, optimize=True)
    s2 += (-0.5) * np.einsum('cdki,abcdjk->abij', t2, I24, optimize=True)
    s2 += (-1.5) * np.einsum('cdkj,abcdik->abij', t2, I23, optimize=True)
    s2 += (0.5) * np.einsum('cdli,abcdjl->abij', r2, I8, optimize=True)
    s2 += (0.5) * np.einsum('cdlj,abcdil->abij', r2, I9, optimize=True)
    s2 += (0.5) * np.einsum('ci,abkl,cjkl->abij', t1, r2, I5, optimize=True)
    s2 += (0.5) * np.einsum('ci,adkj,bcdk->abij', t1, t2, I2, optimize=True)
    s2 += (-1.0) * np.einsum('ci,ak,bcjk->abij', t1, r1, I38, optimize=True)
    s2 += (1.0) * np.einsum('ci,al,bcjl->abij', t1, r1, I42, optimize=True)
    s2 += (1.0) * np.einsum('ci,dbjl,acdl->abij', t1, r2, I1, optimize=True)
    s2 += (0.5) * np.einsum('ci,dbkj,acdk->abij', t1, t2, I3, optimize=True)
    s2 += (-2.0) * np.einsum('ci,dblj,acdl->abij', t1, r2, I1, optimize=True)
    s2 += (-1.0) * np.einsum('ci,dj,abcd->abij', t1, r1, I51, optimize=True)
    s2 += (0.5) * np.einsum('cj,adik,bcdk->abij', t1, t2, I2, optimize=True)
    s2 += (-1.0) * np.einsum('cj,adil,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (1.0) * np.einsum('cj,al,bcil->abij', t1, r1, I77, optimize=True)
    s2 += (-1.0) * np.einsum('cj,bk,acik->abij', t1, r1, I53, optimize=True)
    s2 += (1.0) * np.einsum('cj,bl,acil->abij', t1, r1, I78, optimize=True)
    s2 += (1.0) * np.einsum('cj,dail,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (-1.0) * np.einsum('cj,dali,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (0.5) * np.einsum('cj,dbik,acdk->abij', t1, t2, I3, optimize=True)
    s2 += (-1.0) * np.einsum('cj,di,abcd->abij', t1, r1, I52, optimize=True)
    s2 += (1.0) * np.einsum('cj,di,al,bcdl->abij', t1, t1, r1, I0, optimize=True)
    s2 += (-2.0) * np.einsum('ck,abil,cjkl->abij', t1, r2, I5, optimize=True)
    s2 += (-1.0) * np.einsum('ck,abil,cjkl->abij', t1, t2, I13, optimize=True)
    s2 += (-2.0) * np.einsum('ck,ablj,cikl->abij', t1, r2, I6, optimize=True)
    s2 += (-1.0) * np.einsum('ck,ablj,cikl->abij', t1, t2, I12, optimize=True)
    s2 += (-1.0) * np.einsum('ck,adij,bcdk->abij', t1, t2, I2, optimize=True)
    s2 += (-1.0) * np.einsum('ck,dbij,acdk->abij', t1, t2, I3, optimize=True)
    s2 += (1.0) * np.einsum('cl,adij,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (1.0) * np.einsum('cl,dbij,acdl->abij', t1, r2, I1, optimize=True)
    s2 += (0.5) * np.einsum('cl,di,abcdjl->abij', t1, r1, I8, optimize=True)
    s2 += (0.5) * np.einsum('cl,dj,abcdil->abij', t1, r1, I9, optimize=True)
    s2 += (1.0) * np.einsum('daik,bdjk->abij', t2, I46, optimize=True)
    s2 += (-2.0) * np.einsum('dail,bdjl->abij', r2, I30, optimize=True)
    s2 += (1.0) * np.einsum('dail,bdjl->abij', r2, I31, optimize=True)
    s2 += (0.5) * np.einsum('dail,cj,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (-1.0) * np.einsum('daki,bdjk->abij', t2, I46, optimize=True)
    s2 += (0.5) * np.einsum('dakl,bdijkl->abij', r2, I22, optimize=True)
    s2 += (0.5) * np.einsum('dakl,bdijkl->abij', t2, I49, optimize=True)
    s2 += (2.0) * np.einsum('dali,bdjl->abij', r2, I30, optimize=True)
    s2 += (-1.0) * np.einsum('dali,bdjl->abij', r2, I31, optimize=True)
    s2 += (-0.5) * np.einsum('dali,cj,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (-0.5) * np.einsum('dalk,bdijkl->abij', r2, I22, optimize=True)
    s2 += (-0.5) * np.einsum('dalk,bdijkl->abij', t2, I49, optimize=True)
    s2 += (-1.0) * np.einsum('dbij,ck,acdk->abij', t2, t1, I3, optimize=True)
    s2 += (1.0) * np.einsum('dbij,cl,acdl->abij', t2, r1, I1, optimize=True)
    s2 += (-1.0) * np.einsum('dbik,adjk->abij', r2, I72, optimize=True)
    s2 += (0.5) * np.einsum('dbik,al,djkl->abij', t2, t1, I11, optimize=True)
    s2 += (0.5) * np.einsum('dbik,cj,acdk->abij', t2, t1, I3, optimize=True)
    s2 += (1.0) * np.einsum('dbjk,adik->abij', t2, I67, optimize=True)
    s2 += (1.0) * np.einsum('dbjl,adil->abij', r2, I34, optimize=True)
    s2 += (-1.0) * np.einsum('dbjl,adil->abij', r2, I35, optimize=True)
    s2 += (-1.0) * np.einsum('dbjl,adil->abij', r2, I36, optimize=True)
    s2 += (0.5) * np.einsum('dbjl,ci,acdl->abij', t2, r1, I1, optimize=True)
    s2 += (0.5) * np.einsum('dbkj,ci,acdk->abij', t2, r1, I15, optimize=True)
    s2 += (0.5) * np.einsum('dbkj,ci,acdk->abij', t2, t1, I3, optimize=True)
    s2 += (0.5) * np.einsum('dbkl,adijkl->abij', r2, I25, optimize=True)
    s2 += (0.5) * np.einsum('dbkl,adijkl->abij', t2, I70, optimize=True)
    s2 += (-2.0) * np.einsum('dblj,adil->abij', r2, I34, optimize=True)
    s2 += (2.0) * np.einsum('dblj,adil->abij', r2, I35, optimize=True)
    s2 += (2.0) * np.einsum('dblj,adil->abij', r2, I36, optimize=True)
    s2 += (-1.0) * np.einsum('dblj,ci,acdl->abij', t2, r1, I1, optimize=True)
    s2 += (-1.0) * np.einsum('dblk,adijkl->abij', r2, I25, optimize=True)
    s2 += (-1.0) * np.einsum('dblk,adijkl->abij', t2, I70, optimize=True)
    s2 += (0.5) * np.einsum('dcij,abcd->abij', r2, I27, optimize=True)
    s2 += (-1.0) * np.einsum('dcij,abcd->abij', r2, I52, optimize=True)
    s2 += (-1.0) * np.einsum('dcij,abcd->abij', t2, I76, optimize=True)
    s2 += (1.0) * np.einsum('dcij,al,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (-1.5) * np.einsum('dcik,abcdjk->abij', t2, I24, optimize=True)
    s2 += (0.5) * np.einsum('dcil,abcdjl->abij', r2, I8, optimize=True)
    s2 += (-0.5) * np.einsum('dcjk,abcdik->abij', t2, I23, optimize=True)
    s2 += (0.5) * np.einsum('dcjl,abcdil->abij', r2, I9, optimize=True)
    s2 += (0.5) * np.einsum('dcki,abcdjk->abij', t2, I24, optimize=True)
    s2 += (0.5) * np.einsum('dckj,abcdik->abij', t2, I23, optimize=True)
    s2 += (-0.5) * np.einsum('dcli,abcdjl->abij', r2, I8, optimize=True)
    s2 += (-1.5) * np.einsum('dclj,abcdil->abij', r2, I9, optimize=True)
    s2 += (0.5) * np.einsum('di,ablk,djkl->abij', t1, r2, I4, optimize=True)
    s2 += (1.0) * np.einsum('di,aclj,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (0.5) * np.einsum('di,cbjk,acdk->abij', t1, t2, I3, optimize=True)
    s2 += (-1.0) * np.einsum('di,cbkj,acdk->abij', t1, t2, I3, optimize=True)
    s2 += (1.0) * np.einsum('di,cblj,acdl->abij', t1, r2, I1, optimize=True)
    s2 += (-1.0) * np.einsum('di,cj,abcd->abij', t1, r1, I52, optimize=True)
    s2 += (-0.5) * np.einsum('dj,acik,bcdk->abij', t1, t2, I2, optimize=True)
    s2 += (1.0) * np.einsum('dj,acil,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (0.5) * np.einsum('dj,caik,bcdk->abij', t1, t2, I2, optimize=True)
    s2 += (-0.5) * np.einsum('dj,caki,bcdk->abij', t1, t2, I2, optimize=True)
    s2 += (1.0) * np.einsum('dj,cbil,acdl->abij', t1, r2, I1, optimize=True)
    s2 += (1.0) * np.einsum('dj,ci,bl,acdl->abij', t1, t1, r1, I1, optimize=True)
    s2 += (1.0) * np.einsum('dk,abil,djkl->abij', t1, r2, I4, optimize=True)
    s2 += (0.5) * np.einsum('dk,abil,djkl->abij', t1, t2, I11, optimize=True)
    s2 += (1.0) * np.einsum('dk,ablj,dikl->abij', t1, r2, I7, optimize=True)
    s2 += (0.5) * np.einsum('dk,acij,bcdk->abij', t1, t2, I2, optimize=True)
    s2 += (0.5) * np.einsum('dk,cbij,acdk->abij', t1, t2, I3, optimize=True)
    s2 += (-2.0) * np.einsum('dl,acij,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (-2.0) * np.einsum('dl,cbij,acdl->abij', t1, r2, I1, optimize=True)
    s2 += (-1.0) * np.einsum('dl,ci,abcdjl->abij', t1, r1, I8, optimize=True)
    s2 += (-1.0) * np.einsum('dl,cj,abcdil->abij', t1, r1, I9, optimize=True)
    s2 += (-1.0) * np.einsum('kacd,cbij,dk->abij', get_view('g', 'kacd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kacd,ci,dbkj->abij', get_view('g', 'kacd'), t1, r2, optimize=True)
    s2 += (2.0) * np.einsum('kacd,ck,dbij->abij', get_view('g', 'kacd'), t1, r2, optimize=True)
    s2 += (2.0) * np.einsum('kacd,dbij,ck->abij', get_view('g', 'kacd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kacd,dbik,cj->abij', get_view('g', 'kacd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kacd,dbkj,ci->abij', get_view('g', 'kacd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kacd,dk,cbij->abij', get_view('g', 'kacd'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kaci,bk,cj->abij', get_view('g', 'kaci'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kaci,cbjk->abij', get_view('g', 'kaci'), r2, optimize=True)
    s2 += (2.0) * np.einsum('kaci,cbkj->abij', get_view('g', 'kaci'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kaci,cj,bk->abij', get_view('g', 'kaci'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kaic,cbkj->abij', get_view('g', 'kaic'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kajc,bk,ci->abij', get_view('g', 'kajc'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kajc,cbik->abij', get_view('g', 'kajc'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kajc,ci,bk->abij', get_view('g', 'kajc'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kaji,bk->abij', get_view('g', 'kaji'), r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbcd,acij,dk->abij', get_view('g', 'kbcd'), t2, r1, optimize=True)
    s2 += (2.0) * np.einsum('kbcd,adij,ck->abij', get_view('g', 'kbcd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbcd,adik,cj->abij', get_view('g', 'kbcd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbcd,adkj,ci->abij', get_view('g', 'kbcd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbcd,cj,adik->abij', get_view('g', 'kbcd'), t1, r2, optimize=True)
    s2 += (2.0) * np.einsum('kbcd,ck,adij->abij', get_view('g', 'kbcd'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kbcd,dk,acij->abij', get_view('g', 'kbcd'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('kbcj,acik->abij', get_view('g', 'kbcj'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kbcj,ak,ci->abij', get_view('g', 'kbcj'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbcj,caik->abij', get_view('g', 'kbcj'), r2, optimize=True)
    s2 += (1.0) * np.einsum('kbcj,caki->abij', get_view('g', 'kbcj'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kbcj,ci,ak->abij', get_view('g', 'kbcj'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbic,ackj->abij', get_view('g', 'kbic'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kbic,ak,cj->abij', get_view('g', 'kbic'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbic,cj,ak->abij', get_view('g', 'kbic'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbij,ak->abij', get_view('g', 'kbij'), r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbjc,acik->abij', get_view('g', 'kbjc'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kc,abik,cj->abij', get_view('f', 'kc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kc,abkj,ci->abij', get_view('f', 'kc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kc,acij,bk->abij', get_view('f', 'kc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kc,ak,cbij->abij', get_view('f', 'kc'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kc,bk,acij->abij', get_view('f', 'kc'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kc,cbij,ak->abij', get_view('f', 'kc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kc,ci,abkj->abij', get_view('f', 'kc'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kc,cj,abik->abij', get_view('f', 'kc'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('ki,abkj->abij', get_view('f', 'ki'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kj,abik->abij', get_view('f', 'kj'), r2, optimize=True)
    s2 += (1.0) * np.einsum('klci,abkj,cl->abij', get_view('g', 'klci'), t2, r1, optimize=True)
    s2 += (0.5) * np.einsum('klci,ablk,cj->abij', get_view('g', 'klci'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('klci,ak,cblj->abij', get_view('g', 'klci'), t1, r2, optimize=True)
    s2 += (0.5) * np.einsum('klci,cj,ablk->abij', get_view('g', 'klci'), t1, r2, optimize=True)
    s2 += (-2.0) * np.einsum('klci,ck,ablj->abij', get_view('g', 'klci'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('klcj,abik,cl->abij', get_view('g', 'klcj'), t2, r1, optimize=True)
    s2 += (0.5) * np.einsum('klcj,abkl,ci->abij', get_view('g', 'klcj'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('klcj,ak,cbil->abij', get_view('g', 'klcj'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('klcj,bk,acil->abij', get_view('g', 'klcj'), t1, r2, optimize=True)
    s2 += (0.5) * np.einsum('klcj,ci,abkl->abij', get_view('g', 'klcj'), t1, r2, optimize=True)
    s2 += (-2.0) * np.einsum('klcj,ck,abil->abij', get_view('g', 'klcj'), t1, r2, optimize=True)
    s2 += (-2.0) * np.einsum('klic,abkj,cl->abij', get_view('g', 'klic'), t2, r1, optimize=True)
    s2 += (0.5) * np.einsum('klic,abkl,cj->abij', get_view('g', 'klic'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('klic,ackj,bl->abij', get_view('g', 'klic'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('klic,cbkj,al->abij', get_view('g', 'klic'), t2, r1, optimize=True)
    s2 += (0.5) * np.einsum('klic,cj,abkl->abij', get_view('g', 'klic'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('klic,ck,ablj->abij', get_view('g', 'klic'), t1, r2, optimize=True)
    s2 += (0.5) * np.einsum('klij,abkl->abij', get_view('g', 'klij'), r2, optimize=True)
    s2 += (1.0) * np.einsum('klij,ak,bl->abij', get_view('g', 'klij'), t1, r1, optimize=True)
    s2 += (-2.0) * np.einsum('kljc,abik,cl->abij', get_view('g', 'kljc'), t2, r1, optimize=True)
    s2 += (0.5) * np.einsum('kljc,ablk,ci->abij', get_view('g', 'kljc'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('kljc,acik,bl->abij', get_view('g', 'kljc'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('kljc,cbik,al->abij', get_view('g', 'kljc'), t2, r1, optimize=True)
    s2 += (0.5) * np.einsum('kljc,ci,ablk->abij', get_view('g', 'kljc'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('kljc,ck,abil->abij', get_view('g', 'kljc'), t1, r2, optimize=True)
    s2 += (0.5) * np.einsum('klji,ablk->abij', get_view('g', 'klji'), r2, optimize=True)
    s2 += (1.0) * np.einsum('klji,bk,al->abij', get_view('g', 'klji'), t1, r1, optimize=True)
    outputs['s2'] = s2
    return outputs

def compute_s1(f, g, t1, t2, r1, r2, o, v):
    return compute_outputs(f, g, t1, t2, r1, r2, o, v)['s1']

def compute_s2(f, g, t1, t2, r1, r2, o, v):
    return compute_outputs(f, g, t1, t2, r1, r2, o, v)['s2']
