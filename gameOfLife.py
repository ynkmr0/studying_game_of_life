import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import sys
import numpy as np
from pylab import cm as cm
import matplotlib.pyplot as plt

mod = SourceModule("""
__global__ void step(int *C, int *M)
{
  int count;
  int n_x = blockDim.x*gridDim.x;

  printf("(block,grid) (%d,%d) threadIx,y(%d,%d)\\n", blockDim.x, gridDim.x, threadIdx.x, threadIdx.y);
  //printf("%d\\n", n_x);

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;

  int threadId = j*n_x+i;
  int i_left; int i_right; int j_down; int j_up;

  if(i==0) {
  i_left=n_x-1;
  } else {
  i_left=i-1;
  }

  if(i==n_x-1) {
  i_right=0;
  } else {
  i_right=i+1;
  }

  if(j==0) {
  j_down=n_x-1;
  } else {
  j_down=j-1;
  }

  if(j==n_x-1) {
  j_up=0;
  } else {
  j_up=j+1;
  }


  count = C[j*n_x+i_left] + C[j_down*n_x+i]
    + C[j*n_x+i_right] + C[j_up*n_x+i] + C[j_up*n_x+i_left]
    + C[j_down*n_x+i_right] + C[j_down*n_x+i_left]
    + C[j_up*n_x+i_right];

  //printf("count %d\\n", count);

  printf("(block,grid) (%d,%d) threadIx,y(%d,%d)count(%d)\\n", blockDim.x, gridDim.x, threadIdx.x, threadIdx.y, count);


//Modify matrix M according to the rules B3/S23:
//規則に従って行列mを変更する
//A cell is "Born" if it has exactly 3 neighbours, 
//細胞は、正確に3つの隣人がいる場合に「誕生」します。
//A cell "Survives" if it has 2 or 3 living neighbours; it dies otherwise.
//細胞は、2つまたは3つの生きている隣人がいれば、「生き残っている」 そうでなければ死ぬ

  if(count < 2 || count > 3)
    M[threadId] = 0; // cell dies

  if(count == 2)
    M[threadId] = C[threadId];// cell stays the same

  if(count == 3)
    M[threadId] = 1; // cell either stays alive, or is born
}
""")


#n*nの行列を作成
def random_init(n):
    #np.random.seed(100)
    M=np.zeros((n,n)).astype(np.int32)
    print(n)
    print(M)
    for i in range(n):
        for j in range(n):
            M[j,i]=np.int32(np.random.randint(2))
    return M

#sys.argv[1][2][3]はgameOfLife.pyの引数のインデックス
n=int(sys.argv[1])
n_iter=int(sys.argv[2])
m=int(sys.argv[3])

#スレッドの数は16で固定でブロックの数を決めてる。
n_block=16
n_grid=int(n/n_block)


n=n_block*n_grid

print(n)



#2次元配列を作る関数を作成n*nの行列、値は2値
#numpyarrayの中身はnumpy.int32

C=random_init(n)
print(C)
c1=C[0]
print(type(c1[0]))

#C行列と同じ大きさの行列を生成し値はランダム
M = np.empty_like(C)
print(M)
print(type(M))

#numpy.ndarrayからpycuda.gpuarray.GPUArrayという型に変換している
C_gpu = gpuarray.to_gpu( C )
print(C_gpu)
C_gpu1=C_gpu[0]
print(C_gpu1[0])
print(type(C_gpu1[0]))
M_gpu = gpuarray.to_gpu( M )
print(M_gpu)
print(type(M_gpu))


#cudacコードをfunc変数に代入
func = mod.get_function("step")

for k in range(n_iter):
  print(k)
  func(C_gpu,M_gpu,block=(n_block,n_block,1),grid=(n_grid,n_grid,1))

  C_gpu, M_gpu = M_gpu, C_gpu

print("%d live cells after %d iterations" %(np.sum(C_gpu.get()),n_iter))


#シュミレーション画面の表示
if m==1:
    print(C_gpu)
    print(M_gpu)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    fig.suptitle("Conway's Game of Life Accelerated with PyCUDA")
    while m==1:
      ax.set_title('Number of Iterations = %d'%(n_iter))
      plt.imshow(C_gpu.get(),origin='lower',cmap='Greys',  interpolation='nearest',vmin=0, vmax=1)
      plt.draw()
      plt.pause(.01)

      func(C_gpu,M_gpu,block=(n_block,n_block,1),grid=(n_grid,n_grid,1))
      print(C_gpu)
      print(M_gpu)
      C_gpu, M_gpu = M_gpu, C_gpu
      print("%d live cells after %d iterations" %(np.sum(C_gpu.get()),n_iter))
      n_iter+=1

