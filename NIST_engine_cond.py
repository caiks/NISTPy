#!/usr/bin/env python3

from NISTDev import *
from timeit import default_timer as timer
from sys import stdout, argv

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

def tframe(f,tt):
    reframe = transformsMapVarsFrame
    nn = sdict([(v,f(v)) for v in tvars(tt)])
    return reframe(tt,nn)

def fframe(f,ff):
    return qqff([tframe(f,tt) for tt in ffqq(ff)])

def decompercondrr(ll,uu,aa,kmax,omax,fmax):
    return parametersSystemsHistoryRepasDecomperConditionalFmaxRepa(kmax,omax,fmax,uu,ll,aa)

if __name__ == '__main__':
    t1 = timer()
    print(">>>")
    stdout.flush()

    valency = int(argv[1])
    modelin = argv[2]
    kmax = int(argv[3])
    omax = int(argv[4])
    fmax = int(argv[5])
    model = argv[6]

    print("valency: %d" % valency)
    print("model in: %s" % modelin)
    print("model out: %s" % model)
    print("kmax: %d" % kmax)
    print("omax: %d" % omax)
    print("fmax: %d" % fmax)
    stdout.flush()

    (uu,hr) = nistTrainBucketedIO(valency)

    print("train size: %d" % hrsize(hr))
    stdout.flush()

    digit = VarStr("digit")
    vv = uvars(uu)
    vvl = sset([digit])
    vvk = vv - vvl

    df1 = dfIO(modelin + '.json')

    uu1 = uunion(uu,fsys(dfff(df1)))

    ff1 = fframe(refr1(3),dfnul(uu1,df1,1))

    print("model cardinality: %d" % len(fvars(dfff(df1))))
    stdout.flush()

    uu1 = uunion(uu,fsys(ff1))

    print("underlying level sys cardinality: %d" % len(uvars(uu1)))
    stdout.flush()

    hr1 = hrfmul(uu1,ff1,hr)

    print("underlying level size: %d" % hrsize(hr1))
    stdout.flush()

    print(">>> %s" % model)
    stdout.flush()
    (uu2,df2) = decompercondrr(vvl,uu1,hr1,kmax,omax,fmax)
    df21 = zzdf(funcsTreesMap(lambda xx:(xx[0],fdep(xx[1]|ff1,fder(xx[1]))),dfzz(df2)))
    open(model+".json","w").write(decompFudsPersistentsEncode(decompFudsPersistent(df21)))
    print("<<< done %s" % model)
    stdout.flush()

    print("model cardinality: %d" % len(fvars(dfff(df21))))

    hr1 = hrev([i for i in range(hrsize(hr)) if i % 8 == 0],hr)

    print("train size: %d" % hrsize(hr1))
    stdout.flush()

    pp = treesPaths(hrmult(uu2,df21,hr1))
    bmwrite(model+".bmp",ppbm2(uu,vvk,28,1,2,pp))
    bmwrite(model+"_1.bmp",ppbm(uu,vvk,28,1,2,pp))
    bmwrite(model+"_2.bmp",ppbm(uu,vvk,28,2,2,pp))

    t2 = timer()
    print("<<< done %.3fs" % (t2-t1))
    stdout.flush()
