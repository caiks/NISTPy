#!/usr/bin/env python3

from NISTDev import *
from timeit import default_timer as timer
from sys import stdout
import psutil

if __name__ == '__main__':
    t1 = timer()
    print(">>>")
    stdout.flush()

    (uu,hr) = nistTrainBucketedRectangleRandomIO(2,28,1,17)

    print("train size: %d" % hrsize(hr))
    stdout.flush()

    digit = VarStr("digit")
    vv = uvars(uu)
    vvl = sset([digit])
    vvk = vv - vvl

    model = "NIST_model18_cols"
    (wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed) = (2**11, 8, 2**10, 30, (30*3), 3, 2**8, 1, 127, 1, 5)

    print(">>> %s" % model)
    (uu1,df1) = decomperIO(uu,vvk,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed)
    open(model+".json","w").write(decompFudsPersistentsEncode(decompFudsPersistent(df1)))
    print("<<< done %s" % model)
    stdout.flush()

    print("model cardinality: %d" % len(fvars(dfff(df1))))

    hr1 = hrev([i for i in range(hrsize(hr)) if i % 8 == 0],hr)

    print("train size: %d" % hrsize(hr1))
    stdout.flush()

    (a,ad) = summation(mult,seed,uu1,df1,hr1)
    print("alignment: %.2f" % a)
    print("alignment density: %.2f" % ad)
    stdout.flush()

    (uu,hr) = nistTrainBucketedRectangleRandomIO(2,1,28,17)

    print("train size: %d" % hrsize(hr))
    stdout.flush()

    digit = VarStr("digit")
    vv = uvars(uu)
    vvl = sset([digit])
    vvk = vv - vvl

    model = "NIST_model18_rows"
    (wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed) = (2**11, 8, 2**10, 30, (30*3), 3, 2**8, 1, 127, 1, 5)

    print(">>> %s" % model)
    (uu1,df1) = decomperIO(uu,vvk,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed)
    open(model+".json","w").write(decompFudsPersistentsEncode(decompFudsPersistent(df1)))
    print("<<< done %s" % model)
    stdout.flush()

    print("model cardinality: %d" % len(fvars(dfff(df1))))

    hr1 = hrev([i for i in range(hrsize(hr)) if i % 8 == 0],hr)

    print("train size: %d" % hrsize(hr1))
    stdout.flush()

    (a,ad) = summation(mult,seed,uu1,df1,hr1)
    print("alignment: %.2f" % a)
    print("alignment density: %.2f" % ad)
    stdout.flush()

    t2 = timer()
    print("<<< done %ds %dMB" % (t2-t1, psutil.Process().memory_info().peak_wset // 10**6))
    stdout.flush()
