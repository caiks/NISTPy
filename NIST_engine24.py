#!/usr/bin/env python3

from NISTDev import *
from timeit import default_timer as timer
from sys import stdout

def refr1(k):
    def refr1_f(v):
        if isinstance(v, VarPair):
            (w,i) = v._rep
            if isinstance(w, VarPair):
                (f,l) = w._rep
                if isinstance(f, VarInt):
                    return VarPair((VarPair((VarPair((VarPair((VarInt(k),f)),VarInt(0))),l)),i))
                elif isinstance(f, VarPair):
                    (f1,g) = f._rep
                    return VarPair((VarPair((VarPair((VarPair((VarInt(k),f1)),g)),l)),i))
        return v
    return refr1_f

def refr2(x,y):
    def refr2_f(v):
        if isinstance(v, VarPair):
            (i,j) = v._rep
            if isinstance(i, VarInt) and isinstance(j, VarInt):
                return VarPair((VarInt((x-1)+i),VarInt((y-1)+j)))
        return v
    return refr2_f


def tframe(f,tt):
    reframe = transformsMapVarsFrame
    nn = sdict([(v,f(v)) for v in tvars(tt)])
    return reframe(tt,nn)

def fframe(f,ff):
    return qqff([tframe(f,tt) for tt in ffqq(ff)])

def decomperIO(uu,ff,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed):
    return parametersSystemsHistoryRepasDecomperLevelMaxRollByMExcludedSelfHighestFmaxIORepa(wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed,uu,sdict([((wmax,sset(),ff),emptyTree())]),hr)

if __name__ == '__main__':
    t1 = timer()
    print(">>>")
    stdout.flush()

    (uu,hr) = nistTrainBucketedRegionRandomIO(2,10,17)

    print("train size: %d" % hrsize(hr))
    stdout.flush()

    df1 = dfIO('./NIST_model6.json')

    uu1 = uunion(uu,fsys(dfff(df1)))

    ff1 = fframe(refr1(1),dfnul(uu1,dfred(uu1,df1,hr),1))

    (uu,hr) = nistTrainBucketedIO(2)

    print("train size: %d" % hrsize(hr))
    stdout.flush()

    digit = VarStr("digit")
    vv = uvars(uu)
    vvl = sset([digit])
    vvk = vv - vvl

    gg1 = sset()
    for x in [2,6,10,14,18]:
        for y in [2,6,10,14,18]:
            gg1 |= fframe(refr2(x,y),ff1)

    print("underlying level cardinality: %d" % len(fvars(gg1)))
    stdout.flush()

    uu1 = uunion(uu,fsys(gg1))

    print("underlying level sys cardinality: %d" % len(uvars(uu1)))
    stdout.flush()

    model = "NIST_model24"
    (wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed) = (2**11, 8, 2**10, 30, (30*3), 3, 2**8, 1, 127, 1, 5)

    print(">>> %s" % model)
    (uu2,df2) = decomperIO(uu,gg1,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed)
    open(model+".json","w").write(decompFudsPersistentsEncode(decompFudsPersistent(df2)))
    print("<<< done %s" % model)
    stdout.flush()

    print("model cardinality: %d" % len(fvars(dfff(df2))))

    hr1 = hrev([i for i in range(hrsize(hr)) if i % 8 == 0],hr)

    print("train size: %d" % hrsize(hr1))
    stdout.flush()

    (a,ad) = summation(mult,seed,uu2,df2,hr1)
    print("alignment: %.2f" % a)
    print("alignment density: %.2f" % ad)
    stdout.flush()

    pp = treesPaths(hrmult(uu2,df2,hr1))
    bmwrite(model+".bmp",ppbm2(uu,vvk,28,1,2,pp))
    bmwrite(model+"_1.bmp",ppbm(uu,vvk,28,1,2,pp))
    bmwrite(model+"_2.bmp",ppbm(uu,vvk,28,2,2,pp))

    t2 = timer()
    print("<<< done %ds" % t2-t1)
    stdout.flush()

