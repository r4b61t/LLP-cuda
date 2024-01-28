from new_basis_llp_qaoa.statevector_llp_v2 import StateVectorLLPV2, get_str
import numpy as np
import gc

while True:
    print("test vram")
    Q = np.random.rand(4,6)
    A = np.random.rand(6,6)
    m = StateVectorLLPV2(Q, A, 0.5, 1)
    m.taylor_terms = 8
    m.dt = 0.1
    print("begin caching")
    ha = input("cache ha? \n  >")
    if ha:
        m.ha
    hb = input("cache hb? \n  >")
    if hb:
        m.hb
    ht = input("cache taylor? \n  >")
    if ht:
        m.hb_taylor_terms
    hb = input("cache coarse? \n  >")
    if hb:
        m.ub_coarse_cache

    a= input("delete caches? \n  >")
    if a:
        delattr(m, 'ha')
        delattr(m, 'hb')
        delattr(m, 'hb_taylor_terms')
        delattr(m, 'ub_coarse_cache')
        gc.collect()
    end = input("end? \n  >")
    if end:
        break
