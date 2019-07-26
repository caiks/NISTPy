#!/usr/bin/env python3

from NISTDev import *
from timeit import default_timer as timer
from sys import stdout, argv

def amax(aa):
    ll = aall(norm(trim(aa)))
    ll.sort(key = lambda x: x[1], reverse = True)
    return llaa(ll[:1])

if __name__ == '__main__':
    t1 = timer()
    print(">>>")
    stdout.flush()

    valency = int(argv[1])
    breadth = int(argv[2])
    offset = int(argv[3])
    model = argv[4]

    print("valency: %d" % valency)
    print("breadth: %d" % breadth)
    print("offset: %d" % offset)
    print("model: %s" % model)
    stdout.flush()

    (uu,hrtr) = nistTrainBucketedAveragedIO(valency,breadth,offset)

    hr = hrev([i for i in range(hrsize(hrtr)) if i % 8 == 0],hrtr)

    print("selected train size: %d" % hrsize(hr))
    stdout.flush()

    digit = VarStr("digit")
    vv = uvars(uu)
    vvl = sset([digit])
    vvk = vv - vvl

    df1 = dfIO(model + '.json')

    print("model cardinality: %d" % len(fvars(dfff(df1))))
    stdout.flush()

    uu1 = uunion(uu,fsys(dfff(df1)))

    ff1 = dfnul(uu1,df1,9)

    print("nullable fud cardinality: %d" % len(fvars(ff1)))
    print("nullable fud derived cardinality: %d" % len(fder(ff1)))
    print("nullable fud underlying cardinality: %d" % len(fund(ff1)))
    stdout.flush()

    uu1 = uunion(uu,fsys(ff1))

    hr1 = hrfmul(uu1,ff1,hr)

    print("ff label ent: %.16f" % hrlent(uu1,hr1,fder(ff1),vvl))
    stdout.flush()

    (uu,hrte) = nistTestBucketedAveragedIO(valency,breadth,offset)

    hrq = hrev([i for i in range(hrsize(hrte)) if i % 10 == 0],hrte)

    print("test size: %d" % hrsize(hrq))
    stdout.flush()
    
    hrq1 = hrfmul(uu1,ff1,hrq)

    print("effective size: %d" % int(size(mul(hhaa(hrhh(uu1,hrhrred(hrq1,fder(ff1)))),eff(hhaa(hrhh(uu1,hrhrred(hr1,fder(ff1)))))))))
    stdout.flush()

    print("matches: %d" % len([rr for (_,ss) in hhll(hrhh(uu1,hrhrred(hrq1,fder(ff1)|vvl))) for qq in [single(ss,1)] for rr in [araa(uu1,hrred(hrhrsel(hr1,hhhr(uu1,aahh(red(qq,fder(ff1))))),vvl))] if size(rr) > 0 and size(mul(amax(rr),red(qq,vvl))) > 0]))
    stdout.flush()

    t2 = timer()
    print("<<< done %.3fs" % (t2-t1))
    stdout.flush()
