// Compile as:
// Tesla M2070  nvcc SIA_1D_GPU_v1.cu -arch=compute_13 -code=compute_13,sm_20 --ptxas-options=-v
// TITAN        nvcc SIA_1D_GPU_v1.cu -arch=sm_13 --ptxas-options=-v
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <complex.h>
#include <cufft.h>
#include <time.h>
#include <math.h>
#define GPU_ID 2
#define USE_SINGLE_PRECISION     //Comment this line using "!" if you want to use double precision.  

#ifdef USE_SINGLE_PRECISION
#define DAT     float
#define PRECIS  4
#else
#define DAT     double
#define PRECIS  8
#endif

#define BLOCK_X            32
#define BLOCK_Y            32
#define NB_THREADS         (BLOCK_X*BLOCK_Y)
#define GRID_X             32
#define GRID_Y             32
#define NB_BLOCKS          (GRID_X*GRID_Y)
#define NB_MPS_OF_GPU      15

// maximum overlap in x, y, z direction. x : Vx is nx+1, so it is 1; y: Vy is ny+1, so it is 1; z: Vz is nz+1, so it is 1.
#define MAX_X_OVERLAP      0
#define MAX_Y_OVERLAP      0

////// global variables //////   
// (physics)
const DAT  yr      =   31556926.0;
const DAT  g       =   9.81;
const DAT  ro_i    =   910.0;       //// for the ice code
// const DAT  Ly      =   673733.310;//290432.168;////684374.38;//169151.983;//120940.0;//684374.38;
// const DAT  Lx      =   960802.286;
const DAT  Ly      =   581430.227;//290432.168;////684374.38;//169151.983;//120940.0;//684374.38;
const DAT  Lx      =   802532.526;
const DAT  np      =   3.0;
const DAT gam      =   (1.9e-24)*ro_i*g*ro_i*g*ro_i*g*yr;
const DAT gam2     =   (5.7e-20)*ro_i*g*ro_i*g*ro_i*g*yr;
const DAT PI       =   3.14159265359;

// (numerics)
const int nx   = GRID_X*BLOCK_X - MAX_X_OVERLAP; // -MAX_X_OVERLAP, because we want to have some threads available for all cells of any array, also the ones that are bigger than nx.
const int ny   = GRID_Y*BLOCK_Y - MAX_Y_OVERLAP; // -MAX_Y_OVERLAP, because we want to have some threads available for all cells of any array, also the ones that are bigger than ny.

const int nxi  = GRID_X*BLOCK_X - MAX_X_OVERLAP; // -MAX_X_OVERLAP, because we want to have some threads available for all cells of any array, also the ones that are bigger than nx.
const int nyi  = GRID_Y*BLOCK_Y - MAX_Y_OVERLAP; // -MAX_Y_OVERLAP, because we want to have some threads available for all cells of any array, also the ones that are bigger than ny.
const DAT dx   = Lx/(nxi-1);
const DAT dy   = Ly/(nyi-1);
const int nt   = 550000;

#define min(a,b)                     ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a < _b ? _a : _b; })
#define max(a,b)                     ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b);  _a > _b ? _a : _b; })
#define mod(a,b)                     (a % b)
#define def_sizes(A,nx,ny)  const int sizes_##A[] = {nx,ny};                            
#define      size(A,dim)    (sizes_##A[dim-1])
#define     numel(A)        (size(A,1)*size(A,2))
#define       end(A,dim)    (size(A,dim)-1)
#define     zeros(A,nx,ny)  def_sizes(A,nx,ny);                                            \
                            DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                            for(i=0; i < (nx)*(ny); i++){ A##_h[i]=(DAT)0.0; }             \
                            cudaMalloc(&A##_d      ,numel(A)*sizeof(DAT));                 \
                            cudaMemcpy( A##_d,A##_h,numel(A)*sizeof(DAT),cudaMemcpyHostToDevice);
#define      ones(A,nx,ny)  def_sizes(A,nx,ny);                                            \
                            DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                            for(i=0; i < (nx)*(ny); i++){ A##_h[i]=(DAT)1.0; }             \
                            cudaMalloc(&A##_d      ,numel(A)*sizeof(DAT));                 \
                            cudaMemcpy( A##_d,A##_h,numel(A)*sizeof(DAT),cudaMemcpyHostToDevice);
#define   GPUcopy(A,nx,ny)  def_sizes(A,nx,ny);                                            \
                            DAT *A##_d,*A##_h; A##_h = (DAT*)malloc(numel(A)*sizeof(DAT)); \
                            for(i=0; i < (nx)*(ny); i++){ A##_h[i]=(DAT)A[i]; }             \
                            cudaMalloc(&A##_d      ,numel(A)*sizeof(DAT));                 \
                            cudaMemcpy( A##_d,A##_h,numel(A)*sizeof(DAT),cudaMemcpyHostToDevice);
#define gather(A)           cudaMemcpy( A##_h,A##_d,numel(A)*sizeof(DAT),cudaMemcpyDeviceToHost);
#define free_all(A)         free(A##_h);cudaFree(A##_d);
#define   d_xa(A)           ( A[(ix+1) +  iy   *size(A,1)]  - A[(ix)   +  iy *size(A,1)] )
#define   d_ya(A)           ( A[ ix    + (iy+1)*size(A,1)]  - A[ ix    + (iy)*size(A,1)] )

#define   d_xi(A)           ( A[(ix+1) +  iyi  *size(A,1)]  - A[(ix)   +  iyi*size(A,1)] )
#define   d_yi(A)           ( A[ ixi   + (iy+1)*size(A,1)]  - A[ ixi   + (iy)*size(A,1)] )

#define   d_xa2(A)          ( A[(ix+1) +  iy   *size(A,1)]  - (DAT)2.0*A[ ix   +  iy*size(A,1)]  + A[(ix-1)  +  iy   *size(A,1)] )
#define   d_ya2(A)          ( A[ ix    + (iy+1)*size(A,1)]  - (DAT)2.0*A[ ix   +  iy*size(A,1)]  + A[ ix     + (iy-1)*size(A,1)] )

#define   d_xi2(A)          ( A[(ixi+1)+  iyi   *size(A,1)] - (DAT)2.0*A[ ixi  +  iyi*size(A,1)] + A[(ixi-1) +  iyi   *size(A,1)] )
#define   d_yi2(A)          ( A[ ixi   + (iyi+1)*size(A,1)] - (DAT)2.0*A[ ixi  +  iyi*size(A,1)] + A[ ixi    + (iyi-1)*size(A,1)] )

#define   d_xya2(A)         ((A[(ix+1) + (iy+1)*size(A,1)]  - A[ (ix-1) + (iy+1)*size(A,1)]        \
                            - A[(ix+1) + (iy-1)*size(A,1)]  + A[ (ix-1) + (iy-1)*size(A,1)]) )

#define   d_xyi2(A)         ((A[(ixi+1)+ (iyi+1)*size(A,1)] - A[ (ixi-1) + (iyi+1)*size(A,1)]    \
                            - A[(ixi+1)+ (iyi-1)*size(A,1)] + A[ (ixi-1) + (iyi-1)*size(A,1)]) )

#define    all(A)           ( A[ ix    +  iy   *size(A,1)] )
#define    inn(A)           ( A[ ixi   +  iyi  *size(A,1)] )
#define   in_x(A)           ( A[ ixi   +  iy   *size(A,1)] )
#define   in_y(A)           ( A[ ix    +  iyi  *size(A,1)] )
#define  av_xy(A)           ((A[ ix    +  iy   *size(A,1)] \
                            + A[(ix+1) +  iy   *size(A,1)] \
                            + A[ ix    + (iy+1)*size(A,1)] \
                            + A[(ix+1) + (iy+1)*size(A,1)] )*(DAT)0.25)

#define  loc_max(A)         (max(max(max(A[ ix    +  iy    *size(A,1)],  \
                                         A[(ix+1) +  iy    *size(A,1)]), \
                                         A[ ix    + (iy +1)*size(A,1)]), \
                                         A[(ix+1) + (iy +1)*size(A,1)] ))

#define  loc_max_i(A)       (max(max(max(A[ ixi   +  iyi   *size(A,1)],  \
                                         A[(ixi+1)+  iyi   *size(A,1)]), \
                                         A[ ixi   + (iyi+1)*size(A,1)]), \
                                         A[(ixi+1)+ (iyi+1)*size(A,1)] ))

#define  maskP(A)          ( (A[ ix +  iy*size(A,1)] > (DAT)0.0) ? (DAT) 1.0 : (DAT) 0.0 )        
#define  maskM(A)          ( (A[ ix +  iy*size(A,1)] < (DAT)0.0) ? (DAT) 1.0 : (DAT) 0.0 ) 

#define  maskPi(A)          ( (A[ ixi +  iyi*size(A,1)] > (DAT)0.0) ? (DAT) 1.0 : (DAT) 0.0 )        
#define  maskMi(A)          ( (A[ ixi +  iyi*size(A,1)] < (DAT)0.0) ? (DAT) 1.0 : (DAT) 0.0 ) 


#define  maskP_av(A)       ( (((A[ ix   +  iy   *size(A,1)]                                                 \
                           +   A[(ix+1) +  iy   *size(A,1)]                                                 \
                           +   A[ ix    + (iy+1)*size(A,1)]                                                 \
                           +   A[(ix+1) + (iy+1)*size(A,1)] )*(DAT)0.25) > (DAT)0.0) ? (DAT) 1.0 : (DAT) 0.0 )        
#define  maskM_av(A)       ( (((A[ ix   +  iy   *size(A,1)]                                                 \
                           +   A[(ix+1) +  iy   *size(A,1)]                                                 \
                           +   A[ ix    + (iy+1)*size(A,1)]                                                 \
                           +   A[(ix+1) + (iy+1)*size(A,1)] )*(DAT)0.25) < (DAT)0.0) ? (DAT) 1.0 : (DAT) 0.0 )    

#define select(A,ix,iy)    (  A[ ix +  iy*size(A,1)] )
#define  av_xa(A)          (( A[ ix +  iy*size(A,1)]  + A[(ix+1)  +     iy*size(A,1)]  )*(DAT)0.5)
#define  av_ya(A)          (( A[ ix +  iy*size(A,1)]  + A[ ix     + (iy+1)*size(A,1)]  )*(DAT)0.5)
#define  av_xi(A)          (( A[ ix + iyi*size(A,1)] + A[(ix+1) +     iyi*size(A,1)] )*(DAT)0.5)
#define  av_yi(A)          (( A[ ixi + iy*size(A,1)] + A[ ixi    + (iy+1)*size(A,1)] )*(DAT)0.5)

#define  av_ya_dxa(A)     ((((A[(ix+1) + (iy+1)*nx]  - A[ ix    + (iy+1)*nx])/dx)  + ((A[(ix+1) +  iy   *nx] - A[ ix  + iy*nx])/dx))*(DAT)0.5)
#define  av_xa_dya(A)     ((((A[(ix+1) + (iy+1)*nx]  - A[(ix+1) +  iy   *nx])/dy)  + ((A[ ix    + (iy+1)*nx] - A[ ix  + iy*nx])/dy))*(DAT)0.5)

// participate_a: Test if the thread (ix,iy) has to participate for the following computation of (all) A.
// participate_i: Test if the thread (ix,iy) has to participate for the following computation of (inn) A.
#define participate_a(A)   (ix<size(A,1)   && iy<size(A,2)  )
#define participate_i(A)   (ix<size(A,1)-2 && iy<size(A,2)-2)
// update inner from boundaries
#define bc_no_dx(A)        if (ix==0) select(A,0 ,iy) = select(A,1 ,iy);  if (ix==end(A,1)) select(A,end(A,1),iy      ) = select(A,(end(A,1)-1),iy          )
#define bc_no_dy(A)        if (iy==0) select(A,ix,0 ) = select(A,ix,1 );  if (iy==end(A,2)) select(A,ix      ,end(A,2)) = select(A,ix          ,(end(A,2)-1))
////////// ========== CUDA __max & __min subroutine kernels ========== //////////
#define blockId        (blockIdx.x  +  blockIdx.y *gridDim.x)
#define threadId       (threadIdx.x + threadIdx.y*blockDim.x)
#define isBlockMaster  (threadIdx.x==0 && threadIdx.y==0)
// maxval //
#define block_max_init() DAT __thread_maxval;
#define __thread_max(A)   __thread_maxval=0;                                                                            \
                          if (participate_a(A)){ __thread_maxval = abs(select(A,ix ,iy)); }
__shared__ volatile DAT __block_maxval;
#define __block_max(A)    __thread_max(A);                                                            \
                          if (isBlockMaster){ __block_maxval=0; }                                     \
                          __syncthreads();                                                            \
                          for (int i=0; i < (NB_THREADS); i++){                                       \
                            if (i==threadId){ __block_maxval = max(__block_maxval,__thread_maxval); } \
                            __syncthreads();                                                          \
                          }
///////////////////
#define __MPI_max(A)  __device_max_d<<<grid, block>>>(A##_d, size(A,1),size(A,2), __device_maxval_d); \
                      gather(__device_maxval); device_MAX=(DAT)0;            \
                      for (int i=0; i < (grid.x*grid.y); i++){               \
                         device_MAX = max(device_MAX,__device_maxval_h[i]);  \
                      }          

// (Variables for Multi-GPU applications - we will see this another time)
const int npx=1, npy=1;
const int fdims[]  = {npx,npy};
      int  dims[]  = {1,1};     // Initializations for non parallel version.
      int nprocs=1, me=0;      // Initializations for non parallel version.
      int coords[] = {0,0};   // Initializations for non parallel version.

////// Save functions   //////
void save_info()
{

FILE* fid;
// debug: in fortran function I must also do 'if (me==0)' "around" the WHOLE LINE, i.e. not just one command.
if (me==0){  fid=fopen("000000.nxyz"  , "w"); fprintf(fid,"%d %d %d",PRECIS,nx*fdims[0]*dims[0],ny*fdims[1]*dims[1]);         fclose(fid);}
if (me==0){  fid=fopen("000000.nxyz_l", "w"); fprintf(fid,"%d %d",nx,ny);                                                     fclose(fid);}
if (me==0){  fid=fopen("000000.dims"  , "w"); fprintf(fid,"%d %d %d",nprocs,dims[0],dims[1]);                                 fclose(fid);}
if (me==0){  fid=fopen("000000.fdims" , "w"); fprintf(fid,"%d %d %d",fdims[0]*fdims[1],fdims[0],fdims[1]);                    fclose(fid);}
if (me==0){  fid=fopen("000000.runparams", "w"); fprintf(fid,"%d %d %d %d %f %f",GPU_ID,PRECIS,nx,ny,dx,dy);                  fclose(fid);}
}

void save_coords()
{
char* fname = (char*) malloc( 9*sizeof(char));
char* rm    = (char*) malloc(15*sizeof(char));
FILE* fid;
sprintf(fname,"%.6d.co"  ,me   ); 
sprintf(rm   ,"rm -f %s",fname);
system(rm); fid=fopen(fname          , "w"); fprintf(fid,"%d %d",coords[0],coords[1]);                                         fclose(fid);
}

void save_array(DAT* A, size_t nb_elems, char* A_name)
{  // The name must be of size 2!
char* fname = (char*) malloc( 9*sizeof(char));
char* rm    = (char*) malloc(15*sizeof(char));
FILE* fid;
sprintf(fname,"%.6d.%s" ,me,A_name); 
sprintf(rm   ,"rm -f %s",fname    );
system(rm); fid=fopen(fname          , "wb"); fwrite(A, PRECIS, nb_elems, fid);                                                fclose(fid);
}

void read_data(DAT* A_h, DAT* A_d, int nx,int ny,int nz, char* A_name){
char* bname;
size_t nb_elems = nx*ny*nz;
FILE* fid;
asprintf(&bname, "%.6d.%s", me, A_name); 
fid=fopen(bname, "rb"); // Open file
if (!fid){ fprintf(stderr, "\nUnable to open file %s \n", bname); return; }
fread(A_h, PRECIS, nb_elems, fid); fclose(fid);
cudaMemcpy(A_d, A_h, nb_elems*sizeof(DAT), cudaMemcpyHostToDevice);
printf("Init infos: process %d reads data file  %s (size = %dx%dx%d) \n", me,bname,nx,ny,nz);
free(bname);
}
////////// CUDA subroutines //////////
cudaEvent_t start, stop;
void  init_cuda()
{ 
  cudaSetDevice(GPU_ID); // set the GPU device (count starts at 0)
  cudaDeviceReset();     // Reset device before doing anything
  cudaEventCreate(&start); 
  cudaEventCreate(&stop); 
  struct cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // set L1 to prefered
}

void  tic()
{ 
  cudaDeviceSynchronize(); 
  cudaEventRecord(start, 0); 
  cudaEventSynchronize(start); 
}

float toc()
{
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  float secs = ms/1000.;
  return secs;
}

void  print_mtp(char* name, float secs, size_t bytes){
  double gb = bytes/1024./1024./1024.;
  if(secs < 1e-3) printf("%s took %e seconds (%3.3f GB/s)\n", name, secs, gb/secs);
  else printf("%s took %1.3f seconds (%3.3f GB/s)\n", name, secs, gb/secs);
}

void  clean_cuda(){ 
  cudaError_t ce = cudaGetLastError();
  if(ce != cudaSuccess){ printf("ERROR launching GPU C-CUDA program: %s\n", cudaGetErrorString(ce)); cudaDeviceReset();}
  cudaDeviceReset();
}
////////// Kernels  //////////
__global__ void __device_max_d(DAT*A, const int nx_A,const int ny_A,DAT*__device_maxval){

  // CUDA specific
  def_sizes(A,                 nx_A, ny_A);
  
  block_max_init();

  int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
  int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    /////////////// finds the maxval for each block
  __block_max(A);
  __device_maxval[blockId] = __block_maxval;
}  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////// Initialize, SIA & Mask part  //////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void initialize_loop(DAT* H, DAT* S, DAT* B, DAT* Bhat, DAT* M, DAT* Ela_in, DAT* dt, DAT* tempb, DAT* AS, const DAT gammaT, const DAT qk, const DAT diffus, const DAT c1, const int nxi, const int nyi, const DAT beta, const DAT dt_cut, const DAT M_cut)
  {  
    #define M_abl  ( sqrt((max(all(H)*all(M)/((DAT)3600.0*(DAT)24.0*(DAT)365.0)/diffus, (DAT) 0.0 ))) )

    def_sizes(H,         nxi,   nyi  );
    def_sizes(S,         nxi,   nyi  );
    // def_sizes(B_O,       nxi,   nyi  );
    def_sizes(B,         nxi,   nyi  );
    def_sizes(Bhat,      nxi-1, nyi-1);
    def_sizes(M,         nxi,   nyi  );
    def_sizes(dt,        nxi-2, nyi-2);
    def_sizes(Ela_in,    nxi,   nyi  );
    def_sizes(tempb,     nxi,   nyi  );
    def_sizes(AS,        nxi,   nyi  );

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    // int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    // int iyi = iy+1;                                // shifted iy for computations with inner points of an array
   
    // if (participate_a(H))          all(H)          = (DAT) 0.0;
    // if (participate_a(B))          all(B)          = all(B_O);   
    // if (participate_a(C))          all(C)          = min(all(B0), (DAT) 0.0)*c1; 
    // if (participate_a(Ela_in))     all(Ela_in)     = all(Ela_in); //ELA; 
    // if (participate_a(Ela_old))    all(Ela_old)    = all(Ela_in);  
    if (participate_a(dt))         all(dt)         = dt_cut;
    if (participate_a(S))          all(S)          = all(B) + all(H);   
    if (participate_a(M))          all(M)          = min(beta*(all(S) - all(Ela_in)) , M_cut);
    if (participate_a(tempb))      all(tempb)      = - gammaT*(all(S) - all(Ela_in)) + (qk*all(H)/( M_abl + (DAT) 1e-3))*erf(M_abl/(DAT) 1.14142135);
    if (participate_a(Bhat))       all(Bhat)       = max(max(max(B[ix + iy   *size(B,1)],  B[ix+1 + iy*size(B,1)]),     \
                                                                 B[ix +(iy+1)*size(B,1)]), B[ix+1 +(iy+1) *size(B,1)]);
    if (participate_a(AS))        if ( (all(tempb) < (DAT) 0.0  && all(M) > (DAT) 0.0) )// == 1 )
                                       {
            /*(it % 300 == 0) && */      all(AS)   = max( (DAT) 1.0 + (DAT) 0.6667*atan( all(tempb) ), (DAT) 0.001 );
                                        }  
                                  else 
                                       {
                                         all(AS)   = (DAT) 1.0;
                                       }
    #undef M_abl
  }

__global__ void makeMask(DAT* H, DAT* S, DAT* Ela_in, DAT* Ela_old, DAT* H_in_shape, const int nxi, const int nyi)
  {  

    def_sizes(H,         nxi,   nyi  );
    def_sizes(H_in_shape,nxi,   nyi  );
    def_sizes(Ela_in,    nxi,   nyi  );
    def_sizes(S,         nxi,   nyi  );
    def_sizes(Ela_old,   nxi,   nyi  );

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
     
    if (participate_a(H_in_shape))  if  (all(H)              > (DAT) 0.0)  {
                                         all(H_in_shape)     = (DAT) 1.0;
                                          }  
                                    else {
                                         all(H_in_shape)     = (DAT) 0.0;
                                          }  
    if (participate_a(H_in_shape))  if  (all(S)              > all(Ela_in)) {
                                         all(H_in_shape)     = (DAT) 1.0;
                                          }                                    
    if (participate_a(Ela_old))          all(Ela_old)        = all(Ela_in);
  }


__global__ void SchemePart(DAT* S, DAT* Bhat, DAT* Hbar, DAT* Sx, DAT* Sy, const int nxi, const int nyi, const DAT dx, const DAT dy)
  {  
    #define Dx  ( (DAT) 1.0/dx )
    #define Dy  ( (DAT) 1.0/dy )
 
    def_sizes(S,         nxi,   nyi  );
    def_sizes(Hbar,      nxi-1, nyi-1);
    def_sizes(Bhat,      nxi-1, nyi-1);
    def_sizes(Sx,        nxi-1, nyi-1);
    def_sizes(Sy,        nxi-1, nyi-1);
    // def_sizes(SNorm,     nxi-1, nyi-1);

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    
    if (participate_a(Hbar))       all(Hbar) =(max((DAT) 0.0, S[ix   + iy   *size(S,1)] - Bhat[ix   + iy*size(Bhat,1)]) + \
                                               max((DAT) 0.0, S[ix+1 + iy   *size(S,1)] - Bhat[ix   + iy*size(Bhat,1)]) + \
                                               max((DAT) 0.0, S[ix   +(iy+1)*size(S,1)] - Bhat[ix   + iy*size(Bhat,1)]) + \
                                               max((DAT) 0.0, S[ix+1 +(iy+1)*size(S,1)] - Bhat[ix   + iy*size(Bhat,1)]))*(DAT)  0.25  ;

    if (participate_a(Sx))         all(Sx)   =(max(           S[ix+1 + iy   *size(S,1)] , Bhat[ix   + iy*size(Bhat,1)]) - \
                                               max(           S[ix   + iy   *size(S,1)] , Bhat[ix   + iy*size(Bhat,1)]) + \
                                               max(           S[ix+1 +(iy+1)*size(S,1)] , Bhat[ix   + iy*size(Bhat,1)]) - \
                                               max(           S[ix   +(iy+1)*size(S,1)] , Bhat[ix   + iy*size(Bhat,1)]))*((DAT) 0.5*Dx);

    if (participate_a(Sy))         all(Sy)   =(max(           S[ix   +(iy+1)*size(S,1)] , Bhat[ix   + iy*size(Bhat,1)]) - \
                                               max(           S[ix   + iy   *size(S,1)] , Bhat[ix   + iy*size(Bhat,1)]) + \
                                               max(           S[ix+1 +(iy+1)*size(S,1)] , Bhat[ix   + iy*size(Bhat,1)]) - \
                                               max(           S[ix+1 + iy   *size(S,1)] , Bhat[ix   + iy*size(Bhat,1)]))*((DAT) 0.5*Dy);
    
    // if (participate_a(SNorm))      all(SNorm)=((all(Sx)*all(Sx)      + all(Sy)*all(Sy)));

    #undef Dx
    #undef Dy   
  }

__global__ void DD(DAT* Hbar, DAT* Sx, DAT* Sy, DAT* D, DAT* Bxx, DAT* Bxy, DAT* Byy, DAT* tempb, DAT* AS, const DAT gam, const DAT gam2, const DAT dx, const DAT dy, const int nxi, const int nyi, const DAT pGamma)
  {  
    #define Dx    ( (DAT) 1.0/dx  )
    #define Dy    ( (DAT) 1.0/dy  )

    def_sizes(tempb,     nxi,   nyi  );
    def_sizes(Hbar,      nxi-1, nyi-1);
    def_sizes(D,         nxi-1, nyi-1);
    def_sizes(Sx,        nxi-1, nyi-1);
    def_sizes(Sy,        nxi-1, nyi-1);
    def_sizes(Bxx,       nxi-1, nyi-1);
    def_sizes(Bxy,       nxi-1, nyi-1);
    def_sizes(Byy,       nxi-1, nyi-1);    
    def_sizes(AS,        nxi,   nyi  );

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    int iyi = iy+1;                                // shifted iy for computations with inner points of an array

    // if (participate_a(D))          all(D)    = (gam*all(Hbar)*all(Hbar)*all(Hbar)*all(Hbar)*all(Hbar) +  gam2*(maskP_av(tempb) + (maskM_av(M)))*all(Hbar)*all(Hbar)*all(Hbar))*all(SNorm);
    // if (participate_a(D))          all(D)    = (gam*all(Hbar)*all(Hbar)*all(Hbar)*all(Hbar)*all(Hbar) +  gam2*inn(AS)*all(Hbar)*all(Hbar)*all(Hbar))*all(SNorm);
    if (participate_a(D))          all(D)    = (gam*all(Hbar)*all(Hbar)*all(Hbar)*all(Hbar)*all(Hbar) + gam2*inn(AS)*all(Hbar)*all(Hbar)*all(Hbar))*((all(Sx)*all(Sx) \
                                             + all(Sy) *all(Sy)))*( (DAT) 1.0 / ((DAT) 1.0            +  pGamma*(min  (max ( ( all(Sx)*all(Sx)*all(Bxx)  \
                                             - (DAT)2.0*all(Sx)  *all(Sy)*all(Bxy)                    +  all(Sy)*all(Sy)*all(Byy) )/ ( all(Sx)*all(Sx)      \
                                             + all(Sy) *all(Sy) ) , (DAT) 0.0),  (DAT) 1.0)))) ;

    #undef Dx
    #undef Dy
  }

__global__ void compute_fluxes(DAT* S, DAT* B, DAT* D, DAT* qx, DAT* qy, DAT* dt, const DAT dx, const DAT dy, const int nxi, const int nyi, const DAT dt_cut, const DAT c_stab)
  {  

    def_sizes(S,  nxi,   nyi  );
    def_sizes(B,  nxi,   nyi  );
    def_sizes(D,  nxi-1, nyi-1);
    def_sizes(dt, nxi-2, nyi-2);
    def_sizes(qx, nxi-1, nyi-2);
    def_sizes(qy, nxi-2, nyi-1);
    
    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y

    if (participate_a(qx))     all(qx) = (av_ya(D))*(max(max(B[ix+1 + (iy+1)*size(B,1)],B[ix   + (iy+1)*size(B,1)]),(S[ix+1 +(iy+1)*size(S,1)])) - \
                                                     max(max(B[ix+1 + (iy+1)*size(B,1)],B[ix   + (iy+1)*size(B,1)]),(S[ix   +(iy+1)*size(S,1)])))/dx; 

    if (participate_a(qy))     all(qy) = (av_xa(D))*(max(max(B[ix+1 + (iy+1)*size(B,1)],B[ix+1 +  iy   *size(B,1)]),(S[ix+1 +(iy+1)*size(S,1)])) - \
                                                     max(max(B[ix+1 + (iy+1)*size(B,1)],B[ix+1 +  iy   *size(B,1)]),(S[ix+1 + iy   *size(S,1)])))/dy; 

    if (participate_a(dt))     all(dt) = min(c_stab*min(dx*dx,dy*dy)/((DAT)0.001 + av_xy(D)), dt_cut);
    // if (participate_a(dt))     all(dt) = min(c_stab*min(dx*dx,dy*dy)/((DAT)0.001 + loc_max(D)), dt_cut);
  }

__global__ void ice_thickness(DAT* H, DAT* qx, DAT* qy, DAT* M, DAT* dt, DAT* dq, const DAT dx,const DAT dy,const int nxi, const int nyi, const DAT damp)
  {  

    def_sizes(H,   nxi,   nyi  );
    def_sizes(M,   nxi,   nyi  );
    def_sizes(qx,  nxi-1, nyi-2);
    def_sizes(qy,  nxi-2, nyi-1);
    def_sizes(dt,  nxi-2, nyi-2);
    def_sizes(dq,  nxi-2, nyi-2);
    
    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    int iyi = iy+1;                                // shifted iy for computations with inner points of an array
    
        //0.671
    if (participate_a(dq))     all(dq)   = damp*all(dq) +  d_xa(qx)/dx  + d_ya(qy)/dy + inn(M);
    if (participate_i(H))      inn(H)    = max( inn(H)  +  all(dt)*all(dq),(DAT) 0.0); //
    // if (participate_a(dq))     all(dq)   =                          d_xa(qx)/dx  + (d_ya(qy)/dy) + inn(M);
    // if (participate_i(H))      inn(H)    = max(inn(H)   + all(dt)*((d_xa(qx)/dx) + (d_ya(qy)/dy) + inn(M)),(DAT) 0.0); //
    if (ix < nxi && iy<nyi){   H[ 0      +  iy   *nxi]  = (DAT) 0.0;} //H[  1     +  iy   *nxi];
    if (ix < nxi && iy<nyi){   H[ ix     +  0    *nxi]  = (DAT) 0.0;} //H[ ix     +   1   *nxi];
    if (ix < nxi && iy<nyi){   H[(nxi-1) +  iy   *nxi]  = (DAT) 0.0;} //H[(nxi-2) +  iy   *nxi];
    if (ix < nxi && iy<nyi){   H[ ix     +(nyi-1)*nxi]  = (DAT) 0.0;} //H[ ix     +(nyi-2)*nxi];
  }

///////////////////////////////////////////////////////////////////////////////////
/////////// OPT. Algorithm part   /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
__global__ void dHcalc(DAT* H0, DAT* dH, DAT* H, const int nxi, const int nyi)
  {  

    def_sizes(H,     nxi,   nyi);
    def_sizes(H0,    nxi,   nyi);
    def_sizes(dH,    nxi,   nyi);
    
    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    // int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    // int iyi = iy+1;                                // shifted iy for computations with inner points of an array
    
    if (participate_a(dH))   all(dH)  = fabs(all(H) - all(H0));
  }

__global__ void H0drop(DAT* H0, DAT* H, const int nxi, const int nyi)
  {  

    def_sizes(H,     nxi,   nyi);
    def_sizes(H0,    nxi,   nyi);
    
    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    // int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    // int iyi = iy+1;                                // shifted iy for computations with inner points of an array
    
    if (participate_a(H0))   all(H0)  = all(H);
  }

__global__ void padd(DAT* Z, DAT* H, const int nxp, const int nyp, const int nxi, const int nyi)
  {  

    def_sizes(Z,   nxp, nyp);
    def_sizes(H,   nxi, nyi );

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    // int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    // int iyi = iy+1;                                // shifted iy for computations with inner points of an array
   
    if (ix < nxi && iy<nyi){         Z[(nxp-nxi)/2 + ix + ((nyp-nyi)/2 + iy)*nxp]         = H[ix + iy*nxi];}
    }

__global__ void k_xy(DAT* kx, DAT* ky, const int nxp, const int nyp, const int nh, const DAT dx, const DAT dy)
  {  

    def_sizes(kx,  nh, 1);
    def_sizes(ky,  nyp,1);

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    // int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    // int iyi = iy+1;                                // shifted iy for computations with inner points of an array
   
    if (ix  < nh){                  kx[ix]         = DAT(ix)/DAT(dx*nxp);    }
    if (iy  < nyp/2 +1){            ky[iy]         = DAT(iy)/DAT(dy*nyp);    }
    if (iy  > nyp/2 && iy < nyp){   ky[iy]         =-DAT(nyp-iy)/DAT(dy*nyp);}
  }

__global__ void MultiplyKernel(cufftComplex *output, DAT* kx, DAT* ky, const DAT rhoc, const DAT rhom, const DAT rhoi, const DAT Df, const DAT twopi,const int nxp, const int nyp, const int nh) 
 {
    def_sizes(output,    nh, nyp);
    def_sizes(kx,        nh,   1);
    def_sizes(ky,        nyp,  1);

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
   
   if (ix < nh && iy<nyp){
    // float2 c               = output[ix + iy*nxp];
    output[ix + iy*nh].x   = output[ix + iy*nh].x*(rhoi/rhom)/(Df*( ((twopi*kx[ix])*(twopi*kx[ix]) + (twopi*ky[iy])*(twopi*ky[iy]))*((twopi*kx[ix])*(twopi*kx[ix]) + (twopi*ky[iy])*(twopi*ky[iy])) )/(rhom* (DAT) 9.81) + (DAT) 1.0 );
    }
   if (ix < nh && iy<nyp){
    output[ix + iy*nh].y   = output[ix + iy*nh].y*(rhoi/rhom)/(Df*( ((twopi*kx[ix])*(twopi*kx[ix]) + (twopi*ky[iy])*(twopi*ky[iy]))*((twopi*kx[ix])*(twopi*kx[ix]) + (twopi*ky[iy])*(twopi*ky[iy])) )/(rhom* (DAT) 9.81) + (DAT) 1.0 );
    }
 }


__global__ void scale_and_init(DAT* w, DAT* data, DAT* B, DAT* B_O, const int nxp, const int nyp, const int nxi, const int nyi)
  {  

    def_sizes(data,  nxp,   nyp  );
    def_sizes(w,     nxi,   nyi  );
    def_sizes(B,     nxi,   nyi  );
    def_sizes(B_O,   nxi,   nyi  );
    // def_sizes(Bhat,  nxi-1, nyi-1);

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    // int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    // int iyi = iy+1;                                // shifted iy for computations with inner points of an array
   
    if (ix>-1 && iy>-1 && ix < nxi && iy<nyi){  w[ix + iy*nxi]  =                    data[(nxp-nxi)/2 + ix + ((nyp-nyi)/2 + iy)*nxp]/(nxp*nyp);}
    if (ix>-1 && iy>-1 && ix < nxi && iy<nyi){  B[ix + iy*nxi]  = B_O[ix + iy*nxi] - data[(nxp-nxi)/2 + ix + ((nyp-nyi)/2 + iy)*nxp]/(nxp*nyp);}
    // if (participate_a(Bhat))                         all(Bhat)  = max(max(max( B[ix   + iy   *size(B,1)],  B [ix+1 + iy*size(B,1)]),     \
    //                                                                            B[ix   +(iy+1)*size(B,1)]), B [ix+1 +(iy+1) *size(B,1)]);
  }

__global__ void updateSM(DAT* S, DAT* M, DAT* H, DAT* B, DAT* Ela_in, DAT* tempb, DAT* AS, const DAT beta, const DAT gammaT, const DAT qk, const DAT diffus, const DAT M_cut, const int nxi, const int nyi, const int it)
  {  
    #define M_abl  ( sqrt((max(all(H)*all(M)/((DAT)3600.0*24.0*365.0)/diffus, (DAT) 0.0 ))) )
    def_sizes(H,     nxi,   nyi);
    def_sizes(S,     nxi,   nyi);
    def_sizes(B,     nxi,   nyi);
    def_sizes(M,     nxi,   nyi);    
    def_sizes(Ela_in,nxi,   nyi);
    def_sizes(tempb, nxi,   nyi);
    def_sizes(AS,    nxi,   nyi);

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    
    if (participate_a(S))       all(S)     =             all(B) + all(H);
    if (participate_a(M))       all(M)     =   min(beta*(all(S) - all(Ela_in)) , M_cut);
    if (participate_a(tempb))   all(tempb) = -   gammaT*(all(S) - all(Ela_in)) + qk*all(H)/( M_abl + (DAT) 1e-3)*erf(M_abl/(DAT) 1.14142135);
    if ((it % 100 == 0) && ix>-1 && iy>-1 && ix < nxi && iy<nyi ) if ( (all(tempb) < (DAT) 0.0  && all(M) > (DAT) 0.0) )//== 1 )
                                                 {
            /*(it % 300 == 0) && */                  all(AS)    = max( (DAT) 1.0 + (DAT) 0.6667*atan( all(tempb) ), (DAT) 0.001 );
                                                  }  
                                              else 
                                                 {
                                                     all(AS)    = (DAT) 1.0;
                                                  }  

    #undef M_abl
  }

 __global__ void ElaTau(DAT* Ela_in, DAT* H_m_shape, DAT* H_in_shape, const DAT tau, const int nxi, const int nyi)
  {  

    def_sizes(Ela_in,       nxi,   nyi);
    def_sizes(H_m_shape,    nxi,   nyi);
    def_sizes(H_in_shape,   nxi,   nyi);
    
    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    // int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    // int iyi = iy+1;                                // shifted iy for computations with inner points of an array
    if (participate_a(Ela_in))    all(Ela_in)   = all(Ela_in) - tau*(all(H_m_shape) - all(H_in_shape));
  }

__global__ void ElaDerv(DAT* Ela_in, DAT* Ela_d2, const int nxi, const int nyi, const DAT dx, const DAT dy, const DAT tau_2)
  {  

    def_sizes(Ela_in,       nxi,   nyi  );
    def_sizes(Ela_d2,       nxi-2, nyi-2);
   
    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    int iyi = iy+1;                                // shifted iy for computations with inner points of an array

    
    if (participate_a(Ela_d2))    all(Ela_d2)   = tau_2*(d_xi2(Ela_in)/(dx*dx) + d_yi2(Ela_in)/(dy*dy));
  }

__global__ void ElaDiff(DAT* Ela_in, DAT* Ela_d2, const DAT dt_smooth, const int nxi, const int nyi)
  {  

    def_sizes(Ela_in,          nxi,   nyi  );
    def_sizes(Ela_d2,          nxi-2, nyi-2);

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    int iyi = iy+1;                                // shifted iy for computations with inner points of an array
    
    if (participate_i(Ela_in))    inn(Ela_in)   = inn(Ela_in)   + (all(Ela_d2));
    Ela_in[ 0      +  iy   *nxi]                = Ela_in[  1    +  iy     *nxi];
    Ela_in[ ix     +  0    *nxi]                = Ela_in[ ix    +   1     *nxi];
    Ela_in[(nxi-1) +  iy   *nxi]                = Ela_in[(nxi-2)+  iy     *nxi];
    Ela_in[ ix     +(nyi-1)*nxi]                = Ela_in[ ix    + (nyi-2) *nxi];
  }

__global__ void deltaEla(DAT* Ela_in, DAT* Ela_old, DAT* dEla, DAT* H_m_shape,const int nxi, const int nyi)
  {  

    def_sizes(Ela_in,          nxi,   nyi);
    def_sizes(Ela_old,         nxi,   nyi);
    def_sizes(dEla,            nxi,   nyi);
    def_sizes(H_m_shape,       nxi,   nyi);

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    // int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    // int iyi = iy+1;                                // shifted iy for computations with inner points of an array
    
    if (participate_a(dEla))      all(dEla)     = fabs(all(Ela_old) - all(Ela_in))*all(H_m_shape);
  }  

  __global__ void Constriction_factor(DAT* B, DAT* Bhat, DAT* Bxx, DAT* Bxy, DAT* Byy, const int nxi, const int nyi, const DAT dx, const DAT dy)
  {  
    #define Dx  ( (DAT) 1.0/dx )
    #define Dy  ( (DAT) 1.0/dy )
    #define av_y_dx1(A)     ((((A[(ix+1) + (iy+1)*size(A,1)]  - A[ ix    + (iy+1)*size(A,1)]))  + ((A[(ix+1) +  iy   *size(A,1)] - A[ ix  + iy   *size(A,1)])))*(DAT)0.5*Dx)
    #define av_y_dx2(A)     ((((A[(ixi+1)+ (iy+1)*size(A,1)]  - A[ ixi   + (iy+1)*size(A,1)]))  + ((A[(ixi+1)+  iy   *size(A,1)] - A[ ixi + iy   *size(A,1)])))*(DAT)0.5*Dx)
    #define av_y_dx_1(A)    ((((A[(ix)   + (iy+1)*size(A,1)]  - A[ ix-1  + (iy+1)*size(A,1)]))  + ((A[(ix)   +  iy   *size(A,1)] - A[ ix-1+ iy   *size(A,1)])))*(DAT)0.5*Dx)

    #define av_x_dy1xy(A)   ((((A[(ix+1) + (iy+1)*size(A,1)]  - A[(ix+1) +  iy   *size(A,1)]))  + ((A[ ix    + (iy+1)*size(A,1)] - A[ ix  + iy   *size(A,1)])))*(DAT)0.5*Dy)
    #define av_x_dy2xy(A)   ((((A[(ixi+1)+ (iy+1)*size(A,1)]  - A[(ixi+1)+  iy   *size(A,1)]))  + ((A[ ixi   + (iy+1)*size(A,1)] - A[ ixi + iy   *size(A,1)])))*(DAT)0.5*Dy)
    #define av_x_dy_1xy(A)  ((((A[ ix    + (iy+1)*size(A,1)]  - A[ ix    +  iy   *size(A,1)]))  + ((A[ ix-1  + (iy+1)*size(A,1)] - A[ ix-1+ iy   *size(A,1)])))*(DAT)0.5*Dy)

    #define av_x_dy1(A)     ((((A[(ix+1) + (iy+1)*size(A,1)]  - A[(ix+1) +  iy   *size(A,1)]))  + ((A[ ix    + (iy+1)*size(A,1)] - A[ ix  + iy   *size(A,1)])))*(DAT)0.5*Dy)
    #define av_x_dy2(A)     ((((A[(ix+1) +(iyi+1)*size(A,1)]  - A[(ix+1) +  iyi  *size(A,1)]))  + ((A[ ix    +(iyi+1)*size(A,1)] - A[ ix  + iyi  *size(A,1)])))*(DAT)0.5*Dy)
    #define av_x_dy_1(A)    ((((A[(ix+1) + (iy  )*size(A,1)]  - A[(ix+1) + (iy-1)*size(A,1)]))  + ((A[ ix    +  iy   *size(A,1)] - A[ ix  +(iy-1)*size(A,1)])))*(DAT)0.5*Dy)

    def_sizes(B,         nxi,   nyi  );
    def_sizes(Bhat,      nxi-1, nyi-1);
    def_sizes(Bxx,       nxi-1, nyi-1);
    def_sizes(Bxy,       nxi-1, nyi-1);
    def_sizes(Byy,       nxi-1, nyi-1);

    int ix  = blockIdx.x*blockDim.x + threadIdx.x; // thread ID, dimension x
    int iy  = blockIdx.y*blockDim.y + threadIdx.y; // thread ID, dimension y
    int ixi = ix+1;                                // shifted ix for computations with inner points of an array
    int iyi = iy+1;                                // shifted iy for computations with inner points of an array
    
    if (participate_a(Bhat))                  all(Bhat)         = max(max(max(B[ix   + iy   *size(B,1)],  B[ix+1 + iy*size(B,1)]),     \
                                                                               B[ix   +(iy+1)*size(B,1)]), B[ix+1 + (iy+1) *size(B,1)]);

    if (ix == 0 && iy<nyi-1)              {    Bxx[ix + iy*nxi]  = (av_y_dx2(B)       - av_y_dx1(B))*Dx;    }
    if (ix >  0 && ix < nxi-1 && iy<nyi-1){    Bxx[ix + iy*nxi]  =  d_xi2(B)*Dx*Dx;                     }
    if (ix == nxi-2 && iy<nyi-1)          {    Bxx[ix + iy*nxi]  = (av_y_dx1(B)       - av_y_dx_1(B))*Dx;   }

    if (iy == 0 && ix<nxi-1)              {    Byy[ix + iy*nxi]  = (av_x_dy2(B)       - av_x_dy1(B))*Dy;    }
    if (iy >  1 && iy > nyi-1 && ix<nxi-1){    Byy[ix + iy*nxi]  =  d_yi2(B)*Dy*Dy;                     }
    if (iy == nyi-1 && ix<nxi-1)          {    Byy[ix + iy*nxi]  = (av_x_dy1(B)       - av_x_dy_1(B))*Dy;   }

    if (ix == 0 && iy<nyi-1)              {    Bxy[ix + iy*nxi]  = (av_x_dy2xy(B)     - av_x_dy1xy(B))*Dx;  }
    if (ix >  0 && ix < nxi-1 && iy<nyi-1){    Bxy[ix + iy*nxi]  =  d_xyi2(B)*Dx*Dy*(DAT)0.25;          }
    if (ix == nxi-2 && iy<nyi-1)          {    Bxy[ix + iy*nxi]  = (av_x_dy1xy(B)     - av_x_dy_1xy(B))*Dx; }

    #undef Dx
    #undef Dy
    #undef av_y_dx1
    #undef av_y_dx2
    #undef av_x_dy1
    #undef av_x_dy2
    #undef av_x_dy1xy
    #undef av_x_dy2xy
    #undef av_y_dx_1
    #undef av_x_dy_1
    #undef av_x_dy_1xy
  }


///////////////////////////////////////////////////////////////////////////////////
////////// MAIN ///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
int main()
{
  DAT time2, time3, dt_smooth;
  DAT ELA, beta, M_cut, dt_cut, c_stab, tau, tau_2, c1, H_err, val_E_i, maxdH, sumdH, damp, device_MAX, dH_MAX, params_h, pGamma, gammaT,conduct, qk, q, diffus;
  DAT ym, nu, te, Df, rhom, rhoc, rhoi, PI, twopi;
  unsigned long nx, ny, nxi, nyi, nxp, nyp, nx1, ny1, nh, n_pad, n_smooth, n_fft, n_iter, NN, N=nx*ny, mem=N*sizeof(DAT);
  int i, j, ij, ii, jj, ix, iy, iter, it, itt;
  cufftComplex* output;
  printf("%dx%d,  %1.3f GB, %d iterations\n", nx, ny, 4*mem/1024./1024./1024., nt);
  dim3 grid, grid2, block;
  block.x = BLOCK_X;
  block.y = BLOCK_Y;
  grid.x  = GRID_X;
  grid.y  = GRID_Y;
  grid2.x = 32*10;
  grid2.y = 32*10;
  printf("launching (%dx%d) grid of blocks with size (%dx%d) \n", grid.x, grid.y, block.x, block.y);
  init_cuda();
  srand (time(NULL));


///////////////////////////////////////////////////////////////////////////////////
////////////////////////// Parameters /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
nxi  = (int) 32*32; // -MAX_X_OVERLAP, because we want to have some threads available for all cells of any array, also the ones that are bigger than nx.
nyi  = (int) 32*32; // -MAX_Y_OVERLAP, because we want to have some threads available for all cells of any array, also the ones that are bigger than ny.

n_iter  = 1001;
tau     = 45;//100;
n_smooth= 400;//800;
tau_2   = 0.25*fmin(dx*dx,dy*dy);//(DAT) 1.4336e5;   /// preset for dt_smooth to be a bit less then 0.1, 6.9663e5 would be for dt_sm = 0.15 

////////////////////////// Flex params /////////////////////////////////////////////
n_pad  = 10;        // 
nxp    = n_pad*nxi; // 96;           //  dimensions of padded grid
nyp    = n_pad*nyi;  //128;         //  dimensions of padded grid
nh     = nxp/2 + 1;               // half grid dimension
n_fft  = 1000;                    // do fft every n_fft steps

// g      = 9.81;
PI     = 3.14159265359;
ym     = 100.0e9;     // Youngs Modulus, Pa
nu     = 0.25;       // Poisson's ratio, nondim
te     = 30.0e3;      // elastic thickness, m
rhom   = 3300.0;    // mantle density, kg/m^3
rhoc   = 2700.0;    // crust density, kg/m^3
rhoi   = 910.0;       // "infill" density, kg/m^3, use 0 to compute deflection, rhoc to compute root
twopi  = 2.0*PI;

Df     = ym*(te*te*te)/12.0/(1.0 - (nu*nu));
////////////////////////// Ice params /////////////////////////////////////////////

beta   = (DAT) 0.004;          //////////////// Mass balance gradient
M_cut  = (DAT) 1.0;

c1     = (DAT) 0*27.0;
val_E_i= (DAT) 1400.0;
// ELA    = (DAT) 3100.0;

dt_cut = (DAT) 0.499;
c_stab = (DAT) 0.1234;
damp   = (DAT) 0.435;
H_err  = (DAT) 0.01;

pGamma  = (DAT) 2000.0; //fmin(dx,dy); //(DAT) 2000.0;
gammaT  = (DAT) 0.005;
conduct = (DAT) 2.35;
q       = (DAT) 70e-3;
qk      = (DAT) sqrt(PI/ (DAT) 2.0)*q/conduct;
diffus  = (DAT) conduct/rhoi/(DAT) 2115.0;
// dt_smooth      = 0.2439*fmin(dx*dx/tau_2,dy*dy/tau_2);  // 0.1;
printf("dt smooth %1.8f \n", tau_2);
printf("%dx%d \n %1.8fx%1.8f \n ", nxi, nyi, dx, dy);
///////////////////////////////////////////////////////////////////////////////////
////////////////////////// Initializations  ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
zeros(__device_maxval,grid.x,grid.y);
zeros(H,               nxi,   nyi  );
zeros(S,               nxi,   nyi  );
zeros(D,               nxi-1, nyi-1);
zeros(qx,              nxi-1, nyi-2);
zeros(qy,              nxi-2, nyi-1);
zeros(B,               nxi,   nyi  );
zeros(B_O,             nxi,   nyi  );
zeros(C,               nxi,   nyi  );
zeros(M,               nxi,   nyi  );
zeros(Ela_old,         nxi,   nyi  );
zeros(dt,              nxi-2, nyi-2);
zeros(dq,              nxi-2, nyi-2);
zeros(H0,              nxi,   nyi  );
zeros(dH,              nxi,   nyi  );
zeros(Sx,              nxi-1, nyi-1);
zeros(Sy,              nxi-1, nyi-1);
zeros(SNorm,           nxi-1, nyi-1);
zeros(Bhat,            nxi-1, nyi-1);
zeros(Hbar,            nxi-1, nyi-1);
zeros(Ela_in,          nxi,   nyi  );
zeros(dEla,            nxi,   nyi  );
zeros(Ela_d2,          nxi-2, nyi-2);
zeros(H_m_shape,       nxi,   nyi  );
zeros(H_in_shape,      nxi,   nyi  );
zeros(tempb,           nxi,   nyi  );
zeros(temps,           nxi,   nyi  );
zeros(AS,              nxi,   nyi  );
zeros(Bxx,             nxi-1, nyi-1);
zeros(Bxy,             nxi-1, nyi-1);
zeros(Byy,             nxi-1, nyi-1);

zeros(kx,   nh,    1);
zeros(ky,   nyp,   1);
zeros(w,    nxi,  nyi);
zeros(data, nxp,  nyp);
zeros(Z,    nxp,  nyp);
cudaMalloc((void **) &output, sizeof(cufftComplex) * nh*nyp); 

DAT  sInit[nxi*nyi]; 
DAT  res[n_iter];
DAT  resE[n_iter];
DAT  maxdE[n_iter];
DAT  X[nxi*nyi];     //on cpu
DAT  Y[nxi*nyi];    //on cpu

  // next, add max
  // zeros_h(params_evol,NB_PARAMS,(int)ceil((DAT)nt*(DAT)iterMax/(DAT)nout));
  // zeros_h(params ,NB_PARAMS,1);
  // zeros(__device_maxval ,grid.x,grid.y);
  // DAT device_MAX=0.0;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// LOADING DATA /////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
printf("dx: %1.4e \n", dx);
printf("dy: %1.4e \n", dy);
read_data(H_m_shape_h,H_m_shape_d,nxi,nyi,1,"IE_Alps_changed_shape_1k_r");
read_data(B_h,B_d,nxi,nyi,1,"B_iold");
read_data(B_O_h,B_O_d,nxi,nyi,1,"B_1k_bilin");
read_data(H_h,H_d,nxi,nyi,1,"H_iold");

for (ix=0; ix<(nxi); ix++)
     {
        for (iy=0; iy<(nyi); iy++) 
          {
            X              [(ix) +  iy *nxi]  = ix*dx;  
            Y              [(ix) +  iy *nxi]  = iy*dy; //- Ly   + iy*dy;  
            Ela_in_h       [(ix) +  iy *nxi]  = val_E_i;// + 100.0*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
          }
     } 
cudaMemcpy(Ela_in_d,Ela_in_h,numel(Ela_in)*sizeof(DAT),cudaMemcpyHostToDevice);
// cudaMemcpy(Ela_m_d,Ela_m_h,numel(Ela_m)*sizeof(DAT),cudaMemcpyHostToDevice);
////////////////////////////// FLEX Prep /////////////////////////////////////////////////////
cufftHandle planR2C; 
cufftHandle planC2R; 

cufftPlan2d(&planR2C, nyp, nxp, CUFFT_R2C);
cufftPlan2d(&planC2R, nyp, nxp, CUFFT_C2R);
k_xy                  <<<grid2,block>>>(kx_d,ky_d,nxp,nyp,nh,dx,dy);
padd                  <<<grid,block>>>(Z_d,H_d,nxp,nyp,nxi,nyi);  
cufftExecR2C(planR2C, (cufftReal*) Z_d, output);
MultiplyKernel        <<<grid2,block>>>(output,kx_d,ky_d,rhoc,rhom,rhoi,Df,twopi,nxp,nyp,nh);
cufftExecC2R(planC2R, output, (cufftReal *) data_d);
scale_and_init        <<<grid,block>>>(w_d,data_d,B_d,B_O_d,nxp,nyp,nxi,nyi); 
tic();
////////////////////////////// MAIN LOOP /////////////////////////////////////////////////////
      for(iter = 0; iter < n_iter; iter++)
         {             
              initialize_loop           <<<grid,block>>>(H_d,S_d,B_d,Bhat_d,M_d,Ela_in_d,dt_d,tempb_d,AS_d,gammaT,qk,diffus,c1,nxi,nyi,beta,dt_cut,M_cut);
////////////////////////// SIA //////////////////////////////////////////////////////////////
              for(it = 0; it < nt; it++)
              {    

                H0drop                  <<<grid,block>>>(H0_d,H_d,nxi,nyi);  
                SchemePart              <<<grid,block>>>(S_d,Bhat_d,Hbar_d,Sx_d,Sy_d,nxi,nyi,dx,dy);  
                DD                      <<<grid,block>>>(Hbar_d,Sx_d,Sy_d,D_d,Bxx_d,Bxy_d,Byy_d,tempb_d,AS_d,gam,gam2,dx,dy,nxi,nyi,pGamma);   
                compute_fluxes          <<<grid,block>>>(S_d,B_d,D_d,qx_d,qy_d,dt_d,dx,dy,nxi,nyi,dt_cut,c_stab);
                ice_thickness           <<<grid,block>>>(H_d,qx_d,qy_d,M_d,dt_d,dq_d,dx,dy,nxi,nyi,damp);
                if (it % n_fft == 0) 
                 {
                  padd                  <<<grid,block>>>(Z_d,H_d,nxp,nyp,nxi,nyi);  
                  cufftExecR2C(planR2C, (cufftReal*) Z_d, output);
                  MultiplyKernel        <<<grid2,block>>>(output,kx_d,ky_d,rhoc,rhom,rhoi,Df,twopi,nxp,nyp,nh);
                  cufftExecC2R(planC2R, output, (cufftReal *) data_d);
                  scale_and_init        <<<grid,block>>>(w_d,data_d,B_d,B_O_d,nxp,nyp,nxi,nyi);  
                  Constriction_factor   <<<grid,block>>>(B_d,Bhat_d,Bxx_d,Bxy_d,Byy_d,nxi,nyi,dx,dy);  
                 }
                updateSM                <<<grid,block>>>(S_d,M_d,H_d,B_d,Ela_in_d,tempb_d,AS_d,beta,gammaT,qk,diffus,M_cut,nxi,nyi,it);
                if ( (it > 100000) && (it % 10000 == 0) )
                {                   
                  dHcalc                <<<grid,block>>>(H0_d,dH_d,H_d,nxi,nyi);
                  __device_max_d        <<<grid,block>>>(dH_d,nxi,nyi, __device_maxval_d); 
                  gather(__device_maxval);
                  device_MAX       = (DAT)0.0;         
                  for (int i=0; i < (grid.x*grid.y); i++)
                    {             
                     device_MAX    = max(device_MAX,__device_maxval_h[i]);  
                    }                        
    
                  if (device_MAX   < H_err)
                    {
                      printf("\n it_fin: %d", it); 
                      printf("\n max dH: %1.8f \n", device_MAX);       
                      break ;
                    }
                  }


              }
/////////////////////////// END SIA /////////////////////////////////////////////////////
              makeMask                  <<<grid,block>>>(H_d,S_d,Ela_in_d,Ela_old_d,H_in_shape_d,nxi,nyi);

                 if (iter == 10)
                  {
                    gather(Ela_in);
                    save_array(Ela_in_h,        numel(Ela_in),                "E_i_10");
                    gather(H_in_shape);   
                    save_array(H_in_shape_h,    numel(H_in_shape),          "H_i_s_10");
                    gather(H);
                    save_array(H_h,             numel(H),                       "H_10");
                    gather(dEla);
                    save_array(dEla_h,          numel(dEla),                 "dEla_10");
                    res[iter]   = 0.0;
                    resE[iter]  = 0.0;
                    maxdE[iter] = 0.0;
                    for (ix=0; ix<(nxi); ix++)
                       {
                          for (iy=0; iy<(nyi); iy++) 
                            {
                                  res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
                                  resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                             
                              if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
                                {
                                  maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
                                }
                             }
                        }
                  }
                 if (iter == 150)
                  {
                    gather(Ela_in);
                    save_array(Ela_in_h,        numel(Ela_in),                "E_i_150");
                    gather(H_in_shape);   
                    save_array(H_in_shape_h,    numel(H_in_shape),          "H_i_s_150");
                    gather(H);
                    save_array(H_h,             numel(H),                       "H_150");
                    gather(dEla);
                    save_array(dEla_h,          numel(dEla),                 "dEla_150");
                    res[iter]   = 0.0;
                    resE[iter]  = 0.0;
                    maxdE[iter] = 0.0;
                    for (ix=0; ix<(nxi); ix++)
                       {
                          for (iy=0; iy<(nyi); iy++) 
                            {
                                  res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
                                  resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                             
                              if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
                                {
                                  maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
                                }
                             }
                        }
                  }

                 if (iter == 300)
                  {
                    gather(Ela_in);
                    save_array(Ela_in_h,        numel(Ela_in),                "E_i_300");
                    gather(H_in_shape);   
                    save_array(H_in_shape_h,    numel(H_in_shape),          "H_i_s_300");
                    gather(H);
                    save_array(H_h,             numel(H),                       "H_300");
                    gather(dEla);
                    save_array(dEla_h,          numel(dEla),                 "dEla_300");
                    res[iter]   = 0.0;
                    resE[iter]  = 0.0;
                    maxdE[iter] = 0.0;
                    for (ix=0; ix<(nxi); ix++)
                       {
                          for (iy=0; iy<(nyi); iy++) 
                            {
                                  res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
                                  resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                             
                              if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
                                {
                                  maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
                                }
                             }
                        }
                  }

                 if (iter == 450)
                  {
                    gather(Ela_in);
                    save_array(Ela_in_h,        numel(Ela_in),                "E_i_450");
                    gather(H_in_shape);   
                    save_array(H_in_shape_h,    numel(H_in_shape),          "H_i_s_450");
                    gather(H);
                    save_array(H_h,             numel(H),                       "H_450");
                    gather(dEla);
                    save_array(dEla_h,          numel(dEla),                 "dEla_450");
                    res[iter]   = 0.0;
                    resE[iter]  = 0.0;
                    maxdE[iter] = 0.0;
                    for (ix=0; ix<(nxi); ix++)
                       {
                          for (iy=0; iy<(nyi); iy++) 
                            {
                                  res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
                                  resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                             
                              if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
                                {
                                  maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
                                }
                             }
                        }
                  }

                 if (iter == 600)
                  {
                    gather(Ela_in);
                    save_array(Ela_in_h,        numel(Ela_in),                "E_i_600");
                    gather(H_in_shape);   
                    save_array(H_in_shape_h,    numel(H_in_shape),          "H_i_s_600");
                    gather(H);
                    save_array(H_h,             numel(H),                       "H_600");
                    gather(dEla);
                    save_array(dEla_h,          numel(dEla),                 "dEla_600");
                    res[iter]   = 0.0;
                    resE[iter]  = 0.0;
                    maxdE[iter] = 0.0;
                    for (ix=0; ix<(nxi); ix++)
                       {
                          for (iy=0; iy<(nyi); iy++) 
                            {
                                  res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
                                  resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                             
                              if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
                                {
                                  maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
                                }
                             }
                        }
                  }

                 if (iter == 750)
                  {
                    gather(Ela_in);
                    save_array(Ela_in_h,        numel(Ela_in),                "E_i_750");
                    gather(H_in_shape);   
                    save_array(H_in_shape_h,    numel(H_in_shape),          "H_i_s_750");
                    gather(H);
                    save_array(H_h,             numel(H),                       "H_750");
                    gather(dEla);
                    save_array(dEla_h,          numel(dEla),                 "dEla_750");
                    res[iter]   = 0.0;
                    resE[iter]  = 0.0;
                    maxdE[iter] = 0.0;
                    for (ix=0; ix<(nxi); ix++)
                       {
                          for (iy=0; iy<(nyi); iy++) 
                            {
                                  res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
                                  resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                             
                              if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
                                {
                                  maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
                                }
                             }
                        }
                  }
                 
                 if (iter == 900)
                  {
                    gather(Ela_in);
                    save_array(Ela_in_h,        numel(Ela_in),              "E_i_900");
                    gather(H_in_shape);   
                    save_array(H_in_shape_h,    numel(H_in_shape),        "H_i_s_900");
                    gather(H);
                    save_array(H_h,             numel(H),                     "H_900");
                    gather(dEla);
                    save_array(dEla_h,          numel(dEla),               "dEla_900");
                    res[iter]   = 0.0;
                    resE[iter]  = 0.0;
                    maxdE[iter] = 0.0;
                    for (ix=0; ix<(nxi); ix++)
                       {
                          for (iy=0; iy<(nyi); iy++) 
                            {
                                  res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
                                  resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                             
                              if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
                                {
                                  maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
                                }
                             }
                        }
                  }

                if (iter == n_iter-1)
                  {
                    gather(Ela_in);
                    save_array(Ela_in_h,        numel(Ela_in),                 "E_i_f");
                    gather(H_in_shape);   
                    save_array(H_in_shape_h,    numel(H_in_shape),           "H_i_s_f");
                    gather(H);
                    save_array(H_h,             numel(H),                        "H_i");
                    gather(dEla);
                    save_array(dEla_h,          numel(dEla),                  "dEla_f");                  
                    res[iter]   = 0.0;
                    resE[iter]  = 0.0;
                    maxdE[iter] = 0.0;
                    for (ix=0; ix<(nxi); ix++)
                       {
                          for (iy=0; iy<(nyi); iy++) 
                            {
                                  res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
                                  resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                             
                              if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
                                {
                                  maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
                                }
                             }
                        }
                    }
              
              ElaTau                    <<<grid,block>>>(Ela_in_d,H_m_shape_d,H_in_shape_d,tau,nxi,nyi);
              for(itt = 0; itt < n_smooth; itt++)
                  {
                    ElaDerv             <<<grid,block>>>(Ela_in_d,Ela_d2_d,nxi,nyi,dx,dy,tau_2);
                    ElaDiff             <<<grid,block>>>(Ela_in_d,Ela_d2_d,dt_smooth,nxi,nyi);
                  }
              deltaEla                  <<<grid,block>>>(Ela_in_d,Ela_old_d,dEla_d,H_m_shape_d,nxi,nyi);
              // if (iter % 150==0)
              // {
              //   gather(H_in_shape);
              //   gather(dEla);
              //   res[iter]   = 0.0;
              //   resE[iter]  = 0.0;
              //   maxdE[iter] = 0.0;
              //   for (ix=0; ix<(nxi); ix++)
              //      {
              //         for (iy=0; iy<(nyi); iy++) 
              //           {
              //                 res[iter]  += fabs((H_m_shape_h[(ix) +  iy *nxi] - H_in_shape_h[(ix) +  iy *nxi]));
              //                 resE[iter] += fabs(dEla_h[(ix) +  iy *nxi]);
                         
              //             if (fabs(dEla_h[(ix) +  iy *nxi]) > maxdE[iter])
              //               {
              //                 maxdE[iter] = dEla_h[(ix)  +  iy *nxi];
              //               }
              //            }
              //       }
              //   }
              
         }
  time2 = toc();
///////////////////////////////////////////////////////////////////////////////////
////////////// END OF MAIN LOOP  //////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////
  gather(dH);
                  // maxdH = (DAT)0.0; sumdH = (DAT)0.0;
                  // for (i=0; i<(nxi*nyi); i++)
                  //  {
                  //       if (fabs(dH_h[i]) > maxdH)
                  //        {
                  //          maxdH            = (dH_h[i]);
                  //          sumdH           += (dH_h[i])/fmin(dx,dy);
                  //         }
                  // }
                 
                  //   printf("\n it: %d", it); 
                  //   printf("\n max dH: %1.8f, %1.8f \n", maxdH, sumdH);
  cufftDestroy(planC2R);
  cufftDestroy(planR2C);
  gather(B);   
  gather(S);   
  gather(w);
  gather(M);

  // gather(data);
  // gather(Z);
  // Print performance
  print_mtp("\n SIA 2D v2:", time2, 3*mem*nt); // 4 is the number we read + number we write back (during 1 iteration)

  // Save to harddisk
  save_info(); 
  save_coords();
  save_array(M_h,             numel(M),                        "M");
  save_array(w_h,             numel(w),                      "w_f");
  save_array(B_h,             numel(B),                        "B");
  save_array(S_h,             numel(S),                        "S");
  save_array(res,             n_iter,                      "res_f");
  save_array(resE,            n_iter,                     "resE_f");
  save_array(maxdE,           n_iter,                    "maxdE_f");
  // save_array(dH_h,            numel(dH),                      "dH");
  // save_array(data_h,           nxp*nyp,                     "data");
  // save_array(Z_h,              nxp*nyp,                        "Z");
print_mtp("\n SIA 2D v2:", time2, 3*mem*nt); 
  // clear host memory & clear device memory
  free_all(H);
  free_all(B);
  free_all(B_O);
  free_all(S);
  free_all(Ela_in);
  free_all(data);
  free_all(w);
  free_all(Z);
  free_all(H_in_shape);
  free_all(H_m_shape);
  free_all(M);
  print_mtp("\n SIA 2D v2:", time2, 3*mem*nt); 
  clean_cuda();
  return 0;
}
