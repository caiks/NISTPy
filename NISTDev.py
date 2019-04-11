from AlignmentDevRepa import *
from PIL import Image
import gzip
import struct

def aahr(uu,aa):
    return hhhr(uu,aahh(aa))

def decomperIO(uu,vv,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed):
    return parametersSystemsHistoryRepasDecomperMaxRollByMExcludedSelfHighestFmaxIORepa(wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed,uu,vv,hr)

# nistTrainBucketedIO :: Int -> IO (System, HistoryRepa)

def nistTrainBucketedIO(d):
    def lluu(ll):
        return listsSystem([(v,sset(ww)) for (v,ww) in ll])
    uvals = systemsVarsSetValue
    f = gzip.open('train-images-idx3-ubyte.gz','rb')
    _, z, rows, cols = struct.unpack(">IIII", f.read(16))
    p = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,rows*cols])
    r = p * d // 256
    f = gzip.open('train-labels-idx1-ubyte.gz','rb')
    _ = f.read(8)
    l = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,1])
    h = np.concatenate((np.transpose(l),np.transpose(r)))
    uu = lluu([(VarStr("digit"),[ValInt(i) for i in range(9)])] + [(VarPair((VarInt(x), VarInt(y))), [ValInt(i) for i in range(d)]) for x in range(1,rows+1) for y in range(1,cols+1)])
    vv = list(uvars(uu))
    mvv = sdict([(v,i) for (i,v) in enumerate(vv)])
    mm = sdict([(v,sdict([(w,i) for (i,w) in enumerate(uvals(uu,v))])) for v in vv])
    sh = tuple([len(mm[v]) for v in vv])
    hr = (vv,mvv,sh,h)
    return (uu,hr)

# nistTrainIO :: IO (System, HistoryRepa)

def nistTrainIO():
    return nistTrainBucketedIO(256)


