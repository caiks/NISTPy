#!/usr/bin/env python3

from NISTDev import *
from timeit import default_timer as timer
from sys import stdout

if __name__ == '__main__':
    t1 = timer()
    print(">>>")
    stdout.flush()

    (uu,hrtr) = nistTrainBucketedRegionRandomIO(2,11,17)

    u = stringsVariable("<6,6>")
    hr = hrhrsel(hrtr,aahr(uu,single(llss([(u,ValInt(1))]),1)))

    print("train size: %d" % hrsize(hr))
    stdout.flush()

    digit = VarStr("digit")
    vv = uvars(uu)
    vvl = sset([digit])
    vvk = vv - vvl

    model = "NIST_model10"
    (wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed) = (2**11, 8, 2**10, 30, (30*3), 3, 2**8, 1, 127, 1, 5)

    print(">>> %s" % model)
    (uu1,df) = decomperIO(uu,vvk,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed)
    w = stringsVariable("<<0,1>,1>")
    ff0 = sset([trans(unit(sset([llss([(u,ValInt(0)),(w,ValInt(0))]),llss([(u,ValInt(1)),(w,ValInt(1))])])),sset([w]))])
    ss0 = llss([(w,ValInt(1))])
    df1 = lldf([[(stateEmpty(),ff0),(ss0,ll[0][1])]+ll[1:] for ll in dfll(df)])
    open(model+".json","w").write(decompFudsPersistentsEncode(decompFudsPersistent(df1)))
    print("<<< done %s" % model)
    stdout.flush()

    print("model cardinality: %d" % len(fvars(dfff(df1))))

    hr1 = hrev([i for i in range(hrsize(hr)) if i % 8 == 0],hr)

    print("train size: %d" % hrsize(hr1))
    stdout.flush()

    uu1 = uunion(uu,fsys(dfff(df1)))

    (a,ad) = summation(mult,seed,uu1,df1,hr1)
    print("alignment: %.2f" % a)
    print("alignment density: %.2f" % ad)
    stdout.flush()

    pp = treesPaths(hrmult(uu1,df1,hr1))
    bmwrite(model+".bmp",ppbm2(uu,vvk,11,3,2,pp))
    bmwrite(model+"_1.bmp",ppbm(uu,vvk,11,3,2,pp))
    bmwrite(model+"_2.bmp",ppbm(uu,vvk,11,2*2,2,pp))

    t2 = timer()
    print("<<< done %ds" % t2-t1)
    stdout.flush()
