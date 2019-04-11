from AlignmentDevRepa import *
from PIL import Image
import gzip
import struct

def aahr(uu,aa):
    return hhhr(uu,aahh(aa))

def decomperIO(uu,vv,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed):
    return parametersSystemsHistoryRepasDecomperMaxRollByMExcludedSelfHighestFmaxIORepa(wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed,uu,vv,hr)

def bmempty(sx,sy):
    return np.zeros((sx,sy,3),'uint8')

def bminsert(bm2,ox,oy,bm1):
    bm3 = np.copy(bm1)
    bm3[ox:ox+bm1.shape[0],oy:oy+bm1.shape[1],:] = bm1
    return bm3

def bmmax(bm2,ox,oy,bm1):
    bm3 = np.copy(bm1)
    bm3[ox:ox+bm1.shape[0],oy:oy+bm1.shape[1],:] = np.maximum(bm1,bm2[ox:ox+bm1.shape[0],oy:oy+bm1.shape[1],:])
    return bm3

def bmwrite(file,bm):
    Image.fromarray(bm).save(file)

def hrbm(a,b,c,q,d,hr):
    z = hrsize(hr)
    (_,_,_,rr) = hr
    ar1 = np.sum(rr,axis=1)
    ar2 = ar1 * 255 // (d-1) // z
    ar3 = ar2.reshape([b,b])
    ar4 = np.transpose(np.reshape(ar3[:,:,np.newaxis]*([1]*c),[b,b*c]))
    ar5 = np.transpose(np.reshape(ar4[:,:,np.newaxis]*([1]*c),[b*c,b*c]))
    bm1 = (ar5[:,:,np.newaxis]*([1]*3)).astype(dtype='uint8')
    bm2 = bmempty(a,a)
    return bminsert(bm2,q,q,bm1)

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

# nistTestBucketedIO :: Int -> IO (System, HistoryRepa)

def nistTestBucketedIO(d):
    def lluu(ll):
        return listsSystem([(v,sset(ww)) for (v,ww) in ll])
    uvals = systemsVarsSetValue
    f = gzip.open('t10k-images-idx3-ubyte.gz','rb')
    _, z, rows, cols = struct.unpack(">IIII", f.read(16))
    p = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,rows*cols])
    r = p * d // 256
    f = gzip.open('t10k-labels-idx1-ubyte.gz','rb')
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

def nistTestIO():
    return nistTestBucketedIO(256)


