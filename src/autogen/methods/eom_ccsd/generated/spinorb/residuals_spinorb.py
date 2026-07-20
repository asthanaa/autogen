import numpy as np

AUTOGEN_SPIN_SUMMED = False
AUTOGEN_SPIN_SUMMED_MODE = 'spinorb'
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
    I1 = np.einsum('klcd,dj->cjkl', get_view('g', 'klcd'), t1, optimize=True)
    I2 = np.einsum('klcd,ak->acdl', get_view('g', 'klcd'), t1, optimize=True)
    I3 = np.einsum('klcd,di->cikl', get_view('g', 'klcd'), t1, optimize=True)
    I4 = np.einsum('klcd,ci->dikl', get_view('g', 'klcd'), r1, optimize=True)
    I5 = np.einsum('klcd,cj->djkl', get_view('g', 'klcd'), r1, optimize=True)
    I6 = np.einsum('klcd,al->acdk', get_view('g', 'klcd'), r1, optimize=True)
    I7 = np.einsum('klcd,bl->bcdk', get_view('g', 'klcd'), r1, optimize=True)
    I9 = np.einsum('klcd,balk->abcd', get_view('g', 'klcd'), t2, optimize=True)
    I10 = np.einsum('klcd,dcji->ijkl', get_view('g', 'klcd'), t2, optimize=True)
    I13 = np.einsum('klcd,bajk->abcdjl', get_view('g', 'klcd'), t2, optimize=True)
    I14 = np.einsum('klcd,baik->abcdil', get_view('g', 'klcd'), t2, optimize=True)
    I15 = np.einsum('klcd,dbji->bcijkl', get_view('g', 'klcd'), t2, optimize=True)
    I16 = np.einsum('klcd,daji->acijkl', get_view('g', 'klcd'), t2, optimize=True)
    I20 = np.einsum('klcd,dbjk->bcjl', get_view('g', 'klcd'), t2, optimize=True)
    I21 = np.einsum('klcd,dajk->acjl', get_view('g', 'klcd'), t2, optimize=True)
    I22 = np.einsum('klcd,dbik->bcil', get_view('g', 'klcd'), t2, optimize=True)
    I23 = np.einsum('klcd,daik->acil', get_view('g', 'klcd'), t2, optimize=True)
    I24 = np.einsum('kbcd,ak->abcd', get_view('g', 'kbcd'), t1, optimize=True)
    I25 = np.einsum('kacd,bk->abcd', get_view('g', 'kacd'), t1, optimize=True)
    I26 = np.einsum('kbcd,dj->bcjk', get_view('g', 'kbcd'), t1, optimize=True)
    I27 = np.einsum('kacd,dj->acjk', get_view('g', 'kacd'), t1, optimize=True)
    I28 = np.einsum('jkbc,aj->abck', get_view('g', 'jkbc'), t1, optimize=True)
    I29 = np.einsum('jkbc,ci->bijk', get_view('g', 'jkbc'), t1, optimize=True)
    I30 = np.einsum('kljc,bk->bcjl', get_view('g', 'kljc'), t1, optimize=True)
    I31 = np.einsum('klic,bk->bcil', get_view('g', 'klic'), t1, optimize=True)
    I32 = np.einsum('kljc,ci->ijkl', get_view('g', 'kljc'), t1, optimize=True)
    I33 = np.einsum('klic,cj->ijkl', get_view('g', 'klic'), t1, optimize=True)
    return {
        'I0': I0,
        'I1': I1,
        'I2': I2,
        'I3': I3,
        'I4': I4,
        'I5': I5,
        'I6': I6,
        'I7': I7,
        'I9': I9,
        'I10': I10,
        'I13': I13,
        'I14': I14,
        'I15': I15,
        'I16': I16,
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
    I9 = inter['I9']
    I10 = inter['I10']
    I13 = inter['I13']
    I14 = inter['I14']
    I15 = inter['I15']
    I16 = inter['I16']
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
    outputs = {}
    s1 = zeros_for_output('ai', o, v)
    s1 += (1.0) * np.einsum('ab,bi->ai', get_view('f', 'ab'), r1, optimize=True)
    s1 += (0.5) * np.einsum('bakj,bijk->ai', r2, I29, optimize=True)
    s1 += (-1.0) * np.einsum('bj,ak,bijk->ai', t1, r1, I29, optimize=True)
    s1 += (0.5) * np.einsum('cbik,abck->ai', r2, I28, optimize=True)
    s1 += (1.0) * np.einsum('ci,bk,abck->ai', t1, r1, I28, optimize=True)
    s1 += (-1.0) * np.einsum('ck,bi,abck->ai', t1, r1, I28, optimize=True)
    s1 += (0.5) * np.einsum('jabc,cbij->ai', get_view('g', 'jabc'), r2, optimize=True)
    s1 += (1.0) * np.einsum('jabc,ci,bj->ai', get_view('g', 'jabc'), t1, r1, optimize=True)
    s1 += (-1.0) * np.einsum('jabc,cj,bi->ai', get_view('g', 'jabc'), t1, r1, optimize=True)
    s1 += (-1.0) * np.einsum('jaib,bj->ai', get_view('g', 'jaib'), r1, optimize=True)
    s1 += (-1.0) * np.einsum('jb,aj,bi->ai', get_view('f', 'jb'), t1, r1, optimize=True)
    s1 += (-1.0) * np.einsum('jb,baij->ai', get_view('f', 'jb'), r2, optimize=True)
    s1 += (-1.0) * np.einsum('jb,bi,aj->ai', get_view('f', 'jb'), t1, r1, optimize=True)
    s1 += (-1.0) * np.einsum('ji,aj->ai', get_view('f', 'ji'), r1, optimize=True)
    s1 += (1.0) * np.einsum('jkbc,caij,bk->ai', get_view('g', 'jkbc'), t2, r1, optimize=True)
    s1 += (-0.5) * np.einsum('jkbc,cakj,bi->ai', get_view('g', 'jkbc'), t2, r1, optimize=True)
    s1 += (-0.5) * np.einsum('jkbc,cbij,ak->ai', get_view('g', 'jkbc'), t2, r1, optimize=True)
    s1 += (1.0) * np.einsum('jkbc,cj,baik->ai', get_view('g', 'jkbc'), t1, r2, optimize=True)
    s1 += (-1.0) * np.einsum('jkib,aj,bk->ai', get_view('g', 'jkib'), t1, r1, optimize=True)
    s1 += (-0.5) * np.einsum('jkib,bakj->ai', get_view('g', 'jkib'), r2, optimize=True)
    s1 += (1.0) * np.einsum('jkib,bj,ak->ai', get_view('g', 'jkib'), t1, r1, optimize=True)
    outputs['s1'] = s1
    s2 = zeros_for_output('abij', o, v)
    s2 += (-1.0) * np.einsum('ac,cbji->abij', get_view('f', 'ac'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('ak,bl,ijkl->abij', t1, r1, I32, optimize=True)
    s2 += (1.0) * np.einsum('ak,bl,ijkl->abij', t1, r1, I33, optimize=True)
    s2 += (1.0) * np.einsum('ak,cbil,cjkl->abij', t1, r2, I1, optimize=True)
    s2 += (1.0) * np.einsum('ak,ci,bl,cjkl->abij', t1, t1, r1, I1, optimize=True)
    s2 += (1.0) * np.einsum('al,ci,bcjl->abij', t1, r1, I30, optimize=True)
    s2 += (-1.0) * np.einsum('al,cj,bcil->abij', t1, r1, I31, optimize=True)
    s2 += (-0.5) * np.einsum('al,dcji,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (1.0) * np.einsum('al,di,cj,bcdl->abij', t1, t1, r1, I0, optimize=True)
    s2 += (-1.0) * np.einsum('al,dj,ci,bcdl->abij', t1, t1, r1, I0, optimize=True)
    s2 += (-0.5) * np.einsum('bacd,dcji->abij', get_view('g', 'bacd'), r2, optimize=True)
    s2 += (1.0) * np.einsum('bacd,di,cj->abij', get_view('g', 'bacd'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('bacd,dj,ci->abij', get_view('g', 'bacd'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('baic,cj->abij', get_view('g', 'baic'), r1, optimize=True)
    s2 += (-1.0) * np.einsum('baik,cl,cjkl->abij', t2, r1, I1, optimize=True)
    s2 += (0.5) * np.einsum('baik,dl,djkl->abij', t2, t1, I5, optimize=True)
    s2 += (1.0) * np.einsum('bajc,ci->abij', get_view('g', 'bajc'), r1, optimize=True)
    s2 += (1.0) * np.einsum('bajk,cl,cikl->abij', t2, r1, I3, optimize=True)
    s2 += (-0.5) * np.einsum('bajk,dl,dikl->abij', t2, t1, I4, optimize=True)
    s2 += (0.5) * np.einsum('balk,ci,cjkl->abij', t2, r1, I1, optimize=True)
    s2 += (-0.5) * np.einsum('balk,cj,cikl->abij', t2, r1, I3, optimize=True)
    s2 += (0.25) * np.einsum('balk,ijkl->abij', r2, I10, optimize=True)
    s2 += (-0.5) * np.einsum('balk,ijkl->abij', r2, I32, optimize=True)
    s2 += (0.5) * np.einsum('balk,ijkl->abij', r2, I33, optimize=True)
    s2 += (1.0) * np.einsum('bc,caji->abij', get_view('f', 'bc'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('caik,bcjk->abij', r2, I26, optimize=True)
    s2 += (0.5) * np.einsum('caik,bl,cjkl->abij', t2, r1, I1, optimize=True)
    s2 += (-1.0) * np.einsum('cail,bcjl->abij', r2, I20, optimize=True)
    s2 += (1.0) * np.einsum('cail,bcjl->abij', r2, I30, optimize=True)
    s2 += (-0.5) * np.einsum('cajk,bl,cikl->abij', t2, r1, I3, optimize=True)
    s2 += (1.0) * np.einsum('cajl,bcil->abij', r2, I22, optimize=True)
    s2 += (-1.0) * np.einsum('cajl,bcil->abij', r2, I31, optimize=True)
    s2 += (-0.5) * np.einsum('calk,bcijkl->abij', r2, I15, optimize=True)
    s2 += (1.0) * np.einsum('cbik,acjk->abij', r2, I27, optimize=True)
    s2 += (-0.5) * np.einsum('cbik,al,cjkl->abij', t2, r1, I1, optimize=True)
    s2 += (1.0) * np.einsum('cbil,acjl->abij', r2, I21, optimize=True)
    s2 += (0.5) * np.einsum('cbjk,al,cikl->abij', t2, r1, I3, optimize=True)
    s2 += (-1.0) * np.einsum('cbjl,acil->abij', r2, I23, optimize=True)
    s2 += (0.5) * np.einsum('cblk,acijkl->abij', r2, I16, optimize=True)
    s2 += (-1.0) * np.einsum('ci,ak,bcjk->abij', t1, r1, I26, optimize=True)
    s2 += (1.0) * np.einsum('ci,al,bcjl->abij', t1, r1, I30, optimize=True)
    s2 += (0.5) * np.einsum('ci,balk,cjkl->abij', t1, r2, I1, optimize=True)
    s2 += (1.0) * np.einsum('ci,bk,acjk->abij', t1, r1, I27, optimize=True)
    s2 += (-1.0) * np.einsum('cj,al,bcil->abij', t1, r1, I31, optimize=True)
    s2 += (1.0) * np.einsum('ck,bail,cjkl->abij', t1, r2, I1, optimize=True)
    s2 += (-1.0) * np.einsum('ck,bajl,cikl->abij', t1, r2, I3, optimize=True)
    s2 += (-0.5) * np.einsum('daik,bl,djkl->abij', t2, t1, I5, optimize=True)
    s2 += (-0.5) * np.einsum('daik,cj,bcdk->abij', t2, t1, I7, optimize=True)
    s2 += (0.5) * np.einsum('dail,cj,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (-0.5) * np.einsum('daji,ck,bcdk->abij', t2, t1, I7, optimize=True)
    s2 += (1.0) * np.einsum('daji,cl,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (0.5) * np.einsum('dajk,bl,dikl->abij', t2, t1, I4, optimize=True)
    s2 += (0.5) * np.einsum('dajk,ci,bcdk->abij', t2, t1, I7, optimize=True)
    s2 += (-0.5) * np.einsum('dajl,ci,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (0.5) * np.einsum('dbik,al,djkl->abij', t2, t1, I5, optimize=True)
    s2 += (0.5) * np.einsum('dbik,cj,acdk->abij', t2, t1, I6, optimize=True)
    s2 += (-0.5) * np.einsum('dbil,cj,acdl->abij', t2, r1, I2, optimize=True)
    s2 += (0.5) * np.einsum('dbji,ck,acdk->abij', t2, t1, I6, optimize=True)
    s2 += (-1.0) * np.einsum('dbji,cl,acdl->abij', t2, r1, I2, optimize=True)
    s2 += (-0.5) * np.einsum('dbjk,al,dikl->abij', t2, t1, I4, optimize=True)
    s2 += (-0.5) * np.einsum('dbjk,ci,acdk->abij', t2, t1, I6, optimize=True)
    s2 += (0.5) * np.einsum('dbjl,ci,acdl->abij', t2, r1, I2, optimize=True)
    s2 += (0.5) * np.einsum('dcil,abcdjl->abij', r2, I13, optimize=True)
    s2 += (-0.5) * np.einsum('dcji,abcd->abij', r2, I24, optimize=True)
    s2 += (0.5) * np.einsum('dcji,abcd->abij', r2, I25, optimize=True)
    s2 += (0.25) * np.einsum('dcji,abcd->abij', r2, I9, optimize=True)
    s2 += (-0.5) * np.einsum('dcji,al,bcdl->abij', t2, r1, I0, optimize=True)
    s2 += (0.5) * np.einsum('dcji,bl,acdl->abij', t2, r1, I2, optimize=True)
    s2 += (-0.5) * np.einsum('dcjl,abcdil->abij', r2, I14, optimize=True)
    s2 += (1.0) * np.einsum('di,cajl,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (-1.0) * np.einsum('di,cbjl,acdl->abij', t1, r2, I2, optimize=True)
    s2 += (1.0) * np.einsum('di,cj,abcd->abij', t1, r1, I24, optimize=True)
    s2 += (-1.0) * np.einsum('di,cj,abcd->abij', t1, r1, I25, optimize=True)
    s2 += (-1.0) * np.einsum('dj,cail,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (-1.0) * np.einsum('dj,ci,abcd->abij', t1, r1, I24, optimize=True)
    s2 += (1.0) * np.einsum('dj,ci,abcd->abij', t1, r1, I25, optimize=True)
    s2 += (-1.0) * np.einsum('dj,ci,al,bcdl->abij', t1, t1, r1, I0, optimize=True)
    s2 += (-0.5) * np.einsum('dk,bail,djkl->abij', t1, t2, I5, optimize=True)
    s2 += (0.5) * np.einsum('dk,bajl,dikl->abij', t1, t2, I4, optimize=True)
    s2 += (0.5) * np.einsum('dk,caji,bcdk->abij', t1, t2, I7, optimize=True)
    s2 += (-0.5) * np.einsum('dk,cbji,acdk->abij', t1, t2, I6, optimize=True)
    s2 += (-1.0) * np.einsum('dl,caji,bcdl->abij', t1, r2, I0, optimize=True)
    s2 += (1.0) * np.einsum('dl,cbji,acdl->abij', t1, r2, I2, optimize=True)
    s2 += (-1.0) * np.einsum('kacd,dbik,cj->abij', get_view('g', 'kacd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kacd,dbji,ck->abij', get_view('g', 'kacd'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('kacd,dbjk,ci->abij', get_view('g', 'kacd'), t2, r1, optimize=True)
    s2 += (0.5) * np.einsum('kacd,dcji,bk->abij', get_view('g', 'kacd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kacd,di,cbjk->abij', get_view('g', 'kacd'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('kacd,dk,cbji->abij', get_view('g', 'kacd'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('kaic,bk,cj->abij', get_view('g', 'kaic'), t1, r1, optimize=True)
    s2 += (1.0) * np.einsum('kaic,cbjk->abij', get_view('g', 'kaic'), r2, optimize=True)
    s2 += (1.0) * np.einsum('kaic,cj,bk->abij', get_view('g', 'kaic'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kajc,bk,ci->abij', get_view('g', 'kajc'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kajc,cbik->abij', get_view('g', 'kajc'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kajc,ci,bk->abij', get_view('g', 'kajc'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kaji,bk->abij', get_view('g', 'kaji'), r1, optimize=True)
    s2 += (1.0) * np.einsum('kbcd,daik,cj->abij', get_view('g', 'kbcd'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('kbcd,daji,ck->abij', get_view('g', 'kbcd'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbcd,dajk,ci->abij', get_view('g', 'kbcd'), t2, r1, optimize=True)
    s2 += (-0.5) * np.einsum('kbcd,dcji,ak->abij', get_view('g', 'kbcd'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('kbcd,di,cajk->abij', get_view('g', 'kbcd'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kbcd,dk,caji->abij', get_view('g', 'kbcd'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kbic,ak,cj->abij', get_view('g', 'kbic'), t1, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kbic,cajk->abij', get_view('g', 'kbic'), r2, optimize=True)
    s2 += (-1.0) * np.einsum('kbic,cj,ak->abij', get_view('g', 'kbic'), t1, r1, optimize=True)
    s2 += (1.0) * np.einsum('kbjc,ak,ci->abij', get_view('g', 'kbjc'), t1, r1, optimize=True)
    s2 += (1.0) * np.einsum('kbjc,caik->abij', get_view('g', 'kbjc'), r2, optimize=True)
    s2 += (1.0) * np.einsum('kbjc,ci,ak->abij', get_view('g', 'kbjc'), t1, r1, optimize=True)
    s2 += (1.0) * np.einsum('kbji,ak->abij', get_view('g', 'kbji'), r1, optimize=True)
    s2 += (1.0) * np.einsum('kc,ak,cbji->abij', get_view('f', 'kc'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('kc,baik,cj->abij', get_view('f', 'kc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kc,bajk,ci->abij', get_view('f', 'kc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kc,bk,caji->abij', get_view('f', 'kc'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kc,caji,bk->abij', get_view('f', 'kc'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('kc,cbji,ak->abij', get_view('f', 'kc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kc,ci,bajk->abij', get_view('f', 'kc'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('kc,cj,baik->abij', get_view('f', 'kc'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('ki,bajk->abij', get_view('f', 'ki'), r2, optimize=True)
    s2 += (1.0) * np.einsum('kj,baik->abij', get_view('f', 'kj'), r2, optimize=True)
    s2 += (0.5) * np.einsum('klcd,dalk,cbji->abij', get_view('g', 'klcd'), t2, r2, optimize=True)
    s2 += (-0.5) * np.einsum('klcd,dblk,caji->abij', get_view('g', 'klcd'), t2, r2, optimize=True)
    s2 += (-0.5) * np.einsum('klcd,dcik,bajl->abij', get_view('g', 'klcd'), t2, r2, optimize=True)
    s2 += (0.5) * np.einsum('klcd,dcjk,bail->abij', get_view('g', 'klcd'), t2, r2, optimize=True)
    s2 += (1.0) * np.einsum('klic,ak,cbjl->abij', get_view('g', 'klic'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('klic,bajk,cl->abij', get_view('g', 'klic'), t2, r1, optimize=True)
    s2 += (0.5) * np.einsum('klic,balk,cj->abij', get_view('g', 'klic'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('klic,cajk,bl->abij', get_view('g', 'klic'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('klic,cbjk,al->abij', get_view('g', 'klic'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('klic,ck,bajl->abij', get_view('g', 'klic'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('kljc,ak,cbil->abij', get_view('g', 'kljc'), t1, r2, optimize=True)
    s2 += (1.0) * np.einsum('kljc,baik,cl->abij', get_view('g', 'kljc'), t2, r1, optimize=True)
    s2 += (-0.5) * np.einsum('kljc,balk,ci->abij', get_view('g', 'kljc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kljc,caik,bl->abij', get_view('g', 'kljc'), t2, r1, optimize=True)
    s2 += (1.0) * np.einsum('kljc,cbik,al->abij', get_view('g', 'kljc'), t2, r1, optimize=True)
    s2 += (-1.0) * np.einsum('kljc,ck,bail->abij', get_view('g', 'kljc'), t1, r2, optimize=True)
    s2 += (-1.0) * np.einsum('klji,ak,bl->abij', get_view('g', 'klji'), t1, r1, optimize=True)
    s2 += (-0.5) * np.einsum('klji,balk->abij', get_view('g', 'klji'), r2, optimize=True)
    s2 += (1.0) * np.einsum('klji,bk,al->abij', get_view('g', 'klji'), t1, r1, optimize=True)
    outputs['s2'] = s2
    return outputs

def compute_s1(f, g, t1, t2, r1, r2, o, v):
    return compute_outputs(f, g, t1, t2, r1, r2, o, v)['s1']

def compute_s2(f, g, t1, t2, r1, r2, o, v):
    return compute_outputs(f, g, t1, t2, r1, r2, o, v)['s2']
