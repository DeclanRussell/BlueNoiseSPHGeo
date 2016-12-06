//----------------------------------------------------------------------------------------------------------------------
/// @file SPHSolverCUDAKernals.cu
/// @author Declan Russell
/// @date 03/02/2016
/// @version 1.0
//----------------------------------------------------------------------------------------------------------------------
#include "SPHSolverCUDAKernals.h"
#include <helper_cuda.h>
#include <helper_math.h>  //< some math operations with cuda types
#include <iostream>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

#define NULLHASH 4294967295
#define F_INVTWOPI  ( 0.15915494309f )

// Our simulation properties. These wont change much so lets load them into constant memory
__constant__ SimProps props;


//----------------------------------------------------------------------------------------------------------------------
__global__ void testKernal()
{
    printf("thread number %d\n",threadIdx.x);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void nanCheckKernel(int _n, float3 *_p, int* isNan)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_n)
    {
        if(_p[idx].x!=_p[idx].x||_p[idx].y!=_p[idx].y||_p[idx].z!=_p[idx].z)
        {
            isNan[0] = 1;
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void isValidPos(int _n,int *_convergePtr, float3* _p, float3 *_outP, int* _isValid)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_n)
    {
        float3 p=_p[idx];
        int converged = _convergePtr[idx];
        if(p.x>0.f&&p.x<props.simBounds.x&&p.y>0.f&&p.y<props.simBounds.y&&converged)
        {
            _isValid[idx] = 1.f;
        }
        else
        {
            _isValid[idx] = 0.f;
        }
        _outP[idx]=p;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void fillIntZeroKernal(int *_bufferPtr,int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<size)
    {
        _bufferPtr[idx]=0;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void createHashMapKernal(int _hashTableSize, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_hashTableSize)
    {
        int count=0;
        int key;
        cellInfo cell;
        int z = floor((float)idx/((float)props.gridRes.x*(float)props.gridRes.y));
        int y = floor((float)(idx-(z*props.gridRes.x*props.gridRes.y))/(float)props.gridRes.x);
        int x = idx - ((y*props.gridRes.x)+(z*props.gridRes.x*props.gridRes.y));
        int xi,yj,zk;
        for(int i=-1;i<2;i++)
        {
            for(int j=-1;j<2;j++)
            {
                for(int k=-1;k<2;k++)
                {
                    xi = x+i;
                    yj = y+j;
                    zk = z+k;
                    if ((xi>=0) && (xi<props.gridRes.x) && (yj>=0) && (yj<props.gridRes.y) && (zk>=0) && (zk<props.gridRes.z))
                    {
                        key = idx + i + (j*props.gridRes.x) + (k*props.gridRes.x*props.gridRes.y);
                        if(key>=0 && key<_hashTableSize && count < _hashTableSize)
                        {
                            cell.cIdx[count] = key;
                            count++;
                        }
                    }
                }
            }
        }
        cell.cNum = count;
        _buff.hashMap[idx] = cell;
    }
}

//----------------------------------------------------------------------------------------------------------------------
__device__ int hashPos(float3 _p)
{
    return floor((_p.x/props.gridDim.x)*props.gridRes.x) + (floor((_p.y/props.gridDim.y)*props.gridRes.y)*props.gridRes.x) + (floor((_p.z/props.gridDim.z)*props.gridRes.z)*props.gridRes.x*props.gridRes.y);
}
//----------------------------------------------------------------------------------------------------------------------
__device__ bool isNan(float3 _p)
{
    return (_p.x!=_p.x || _p.y!=_p.y || _p.z!=_p.z);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void hashParticles(int _numParticles, int* _hashKeys, int* _cellOcc,float3* _posPtr)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        float3 pos = (_posPtr[idx] - props.gridMin);
        if(isNan(pos))
        {
            _hashKeys[idx] = NULLHASH;
            printf("NULL HASH idx %d pos %f,%f,%f\n",idx,pos.x,pos.y,pos.z);
        }
        // Make sure the point is within our hash table
        if(pos.x>=0.f && pos.x<props.gridDim.x && pos.y>=0.f && pos.y<props.gridDim.y && pos.z>=0.f && pos.z<props.gridDim.z)
        {
            //Compute our hash key
            int key = hashPos(pos);
            _hashKeys[idx] = key;

            //Increment our occumpancy of this hash cell
            atomicAdd(&(_cellOcc[key]), 1);
        }
        else
        {
            _hashKeys[idx] = NULLHASH;
            printf("NULL HASH idx %d pos %f,%f,%f\n",idx,pos.x,pos.y,pos.z);
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float calculatePressure(float _pi,float _restDensity)
{
    return props.k*(_pi-_restDensity);
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float calcDensityWeighting(float _rLength)
{
    if(_rLength>0.f && _rLength<props.h)
    {
        return props.dWConst * (props.hSqrd - _rLength*_rLength) * (props.hSqrd - _rLength*_rLength) * (props.hSqrd - _rLength*_rLength);
    }
    else
    {
        return 0.f;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float3 calcPressureWeighting(float3 _r, float _rLength)
{
    if(_rLength>0.f && _rLength<props.h)
    {
        return props.pWConst * (_r) * (props.h - _rLength) * (props.h - _rLength);
    }
    else
    {
        return make_float3(0.f,0.f,0.f);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float3 calcViscosityWeighting(float3 _r, float _rLength)
{
    if(_rLength>0.f && _rLength<=props.h)
    {
        return props.vWConst * _r * (props.h - _rLength);
    }
    else
    {
        return make_float3(0.f,0.f,0.f);
    }
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float getPixelVal(float3 _p, float* _buff)
{
    float2 np = make_float2(_p.x,_p.y);
    np.x/=props.simBounds.x;
    np.y/=props.simBounds.y;
    if(np.x<0.f || np.x>1.f || np.y<0.f || np.y>1.f)
    {
        printf("out of bounds %f,%f,%f\n",_p.x,_p.y,_p.z);
        return 0.f;
    }

    // get our 4 surrounding pixel corrdinates
    int2 mmin,mmax;
    np*=make_float2(props.imgRes.x-1,props.imgRes.y-1);
    float2 npf = floorf(np);
    if(np.x-npf.x>0.5f)
    {
       mmin.x = (int)npf.x;
       mmax.x = mmin.x+1;
    }
    else
    {
        mmax.x = (int)npf.x;
        mmin.x = mmax.x-1;
    }
    if(np.y-npf.y>0.5f)
    {
       mmin.y = (int)npf.y;
       mmax.y = mmin.y+1;
    }
    else
    {
        mmax.y = (int)npf.y;
        mmin.y = mmax.y-1;
    }
    clamp(mmin,make_int2(0,0),make_int2(props.imgRes.x-1,props.imgRes.y-1));
    clamp(mmax,make_int2(0,0),make_int2(props.imgRes.x-1,props.imgRes.y-1));

    float Q11 = _buff[mmin.x + (mmin.y*props.imgRes.x)];
    float Q12 = _buff[mmax.x + (mmin.y*props.imgRes.x)];
    float Q21 = _buff[mmin.x + (mmax.y*props.imgRes.x)];
    float Q22 = _buff[mmax.x + (mmax.y*props.imgRes.x)];

    float fx1 = lerp(Q11,Q12,np.x-mmin.x);
    float fx2 = lerp(Q21,Q22,np.x-mmin.x);

    //return clamp(_buff[(int)np.x + (int)(np.y*props.outRes.x)]*10.f,0.f,1.f);
    return clamp(lerp(fx1,fx2,np.y-mmin.y),0.f,1.f);
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float sizeFunc(float3 _p, float *imgPtr)
{

    //return 1.f/(((getPixelVal(_p,imgPtr))*props.sfWidth.y)+props.sfWidth.x);

    return 1.f;
}
//----------------------------------------------------------------------------------------------------------------------
__device__ float adpScaler(float _rLength, float _scaleI, float _scaleN)
{
    float s = (2.f*_rLength)/(_scaleI+_scaleN);
    if(s!=s)
    {
        printf("Nan in Den spi %f spn %f _rLength %f pi %f,%f,%f pn %f,%f,%f\n",_scaleI,_scaleN,_rLength);
    }
//    printf("yi %f si %f yj %f sj %f s %f rLength %f\n",_pi.y,sizeFunc(_pi),_pn.y,sizeFunc(_pn),s,_rLength);

    return s;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveDensityKernal(int _numParticles, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get our particle position
        float3 pos = _buff.posPtr[idx];
        int key = hashPos(pos-props.gridMin);
        // Get our neighbouring cell locations for this particle
        cellInfo nCells = _buff.hashMap[key];

        // Compute our density for all our particles
        float3 pi = pos;
        float di = 0.f;
        float si = sizeFunc(pi,_buff.imgPtr);
        for(int c=0; c<nCells.cNum; c++)
        {
            int nIdx;
            float rLength;
            float3 pj;
            // Get our cell occupancy total and start index
            int cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
            int cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
            for(int i=0; i<cellOcc; i++)
            {
                //Get our neighbour particle index
                nIdx = cellIdx+i;
                //Dont want to compare against same particle
                if(nIdx==idx) continue;
                if(nIdx>_numParticles){printf("%d\n",nIdx);continue;}
                // Get our neighbour position
                pj = _buff.posPtr[nIdx];
                //Calculate our arc length
                rLength = length(pi-pj);
                rLength = adpScaler(rLength,si,sizeFunc(pj,_buff.imgPtr));
                //Increment our density
                di+=props.mass*calcDensityWeighting(rLength);
            }
        }
        _buff.denPtr[idx] = di;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveForcesKernal(int _numParticles, float _restDensity, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get our particle position
        float3 pos = _buff.posPtr[idx];
        float3 pi = pos;
        float3 acc = make_float3(0.f,0.f,0.f);
        float di = _buff.denPtr[idx];
        float avgLen = 0.f;
        float s = sizeFunc(pi,_buff.imgPtr);
        float numN = 0;
        // Put this in its own scope means we get some registers back at the end of it (I think)
        if(di>0.f)
        {
            // Get our neighbouring cell locations for this particle
            cellInfo nCells = _buff.hashMap[hashPos(pos-props.gridMin)];

            // Compute our fources for all our particles

            float3 presForce = make_float3(0.f,0.f,0.f);
            for(int c=0; c<nCells.cNum; c++)
            {
                float3 r,w,pj;
                float presi;
                // Get our cell occupancy total and start index
                int cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
                int cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
                for(int i=0; i<cellOcc; i++)
                {
                    //Get our neighbour particle index
                    int nIdx = cellIdx+i;
                    //Dont want to compare against same particle
                    if(nIdx==idx) continue;
                    // Get our neighbour density
                    float dj = _buff.denPtr[nIdx];
                    // Get our neighbour position
                    pj = _buff.posPtr[nIdx];
                    //Get our vector beteen points
                    r = pi - pj;
                    //Calculate our arc length
                    float rLength=length(r);
                    // Normalise our differential
                    r/=length(r);

                    //Compute our particles pressure
                    presi = calculatePressure(di,_restDensity);
                    float presj = calculatePressure(dj,_restDensity);

                    //Weighting
                    //float3 w = calcPressureWeighting(r,rLength);
                    w = calcPressureWeighting(r,adpScaler(rLength,s,sizeFunc(pj,_buff.imgPtr)));

                    // Accumilate our pressure force
                    if(dj>0.f)
                    {
                        //n-=(props.mass/presj)*w;
                        float3 ptemp = ((presi/(di*di)) + (presj/(dj*dj))) * props.mass * w;
                        presForce+= ptemp;
                        avgLen+=rLength;
                        numN+=1.f;
                    }
                }
            }
            // Compute our average distance between neighbours
            avgLen/=numN;

            // Complete our pressure force term
            presForce*=-props.mass;
            //if(pi.y==props.sphere.y) presForce.y=0.f;//-= n*(dot(n,presForce));

            acc = presForce/props.mass;
        }

        // Acceleration limit
        if(dot(acc,acc)>props.accLimit2)
        {
            acc *= props.accLimit/length(acc);
        }

        // Now lets integerate our acceleration using leapfrog to get our new position
        float3 halfBwd = _buff.velPtr[idx] - 0.5f*props.timeStep*acc;
        float3 halfFwd = halfBwd + props.timeStep*acc;
        // Apply velocity dampaning
        halfFwd *= 0.9f;

        //Velocity Limit
        if(dot(halfFwd,halfFwd)>props.velLimit2)
        {
            halfFwd *= props.velLimit/length(halfFwd);
        }


        // Update our position
        float3 oldPos = pos;
        pos+= props.timeStep * halfFwd * s;


        //Place our particles back on our sphere
        //this could potentially have problems with velocity and such not being
        //adjusted but we will leave that for future work (Hes says...)
        if(pos.x<0.f){
            pos.x = 0.f;
        }
        if(pos.y<0.f){
            pos.y = 0.f;
        }
        if(pos.z<0.f){
            pos.z = 0.f;
        }
        if(pos.x>props.simBounds.x){
            pos.x = props.simBounds.x;
        }
        if(pos.y>props.simBounds.y){
            pos.y = props.simBounds.y;
        }
        if(pos.z>props.simBounds.z){
            pos.z = props.simBounds.z;
        }

        _buff.newPosPtr[idx] = pos;

        // Update our velocity
        _buff.velPtr[idx] = halfFwd;

        // Check to see if we have met our converged state
        if(length(oldPos-pos)<props.convergeValue*avgLen||!numN)
        {
            _buff.convergedPtr[idx] = 1;
        }
        else
        {
            _buff.convergedPtr[idx] = 0;
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void solveDensityNoScaleKernal(int _numParticles, float* outBuff, fluidBuffers _buff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_numParticles)
    {
        // Get our particle position
        float3 pos = _buff.posPtr[idx];
        int key = hashPos(pos-props.gridMin);
        // Get our neighbouring cell locations for this particle
        cellInfo nCells = _buff.hashMap[key];

        // Compute our density for all our particles
        float3 pi = pos;
        float di = 0.f;
        for(int c=0; c<nCells.cNum; c++)
        {
            int nIdx;
            float rLength;
            float3 pj;
            // Get our cell occupancy total and start index
            int cellOcc = _buff.cellOccBuffer[nCells.cIdx[c]];
            int cellIdx = _buff.cellIndexBuffer[nCells.cIdx[c]];
            for(int i=0; i<cellOcc; i++)
            {
                //Get our neighbour particle index
                nIdx = cellIdx+i;
                //Dont want to compare against same particle
                if(nIdx==idx) continue;
                if(nIdx>_numParticles){printf("%d\n",nIdx);continue;}
                // Get our neighbour position
                pj = _buff.posPtr[nIdx];
                //Calculate our arc length
                rLength = length(pi-pj);
                //Increment our density
                di+=props.mass*calcDensityWeighting(rLength);
            }
        }
        outBuff[idx] = di;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void arrayMultiply(int _n, float *_ptr, float _val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_n)
    {
        _ptr[idx] *= _val;
    }
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void mergePosAndPDF(int _n, float3 *_pos, float* _pdf)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<_n)
    {
        _pos[idx].z = _pdf[idx];
    }
}
//----------------------------------------------------------------------------------------------------------------------
void test(){
    printf("calling\n");
    testKernal<<<1,1000>>>();
    //make sure all our threads are done
    cudaThreadSynchronize();
    printf("called\n");
}
//----------------------------------------------------------------------------------------------------------------------
float computeAverageDensity(int _numParticles, fluidBuffers _buff)
{
    // Turn our density buffer pointer into a thrust iterater
    thrust::device_ptr<float> t_denPtr = thrust::device_pointer_cast(_buff.denPtr);

    // Use reduce to sum all our densities
    float sum = thrust::reduce(t_denPtr, t_denPtr+_numParticles, 0.f, thrust::plus<float>());

    // Return our average density
    return sum/(float)_numParticles;
}
//----------------------------------------------------------------------------------------------------------------------
void updateSimProps(SimProps *_props)
{
    #ifdef CUDA_42
        // Unlikely we will ever use CUDA 4.2 but nice to have it in anyway I guess?
        cudaMemcpyToSymbol ( "props", _props, sizeof(SimProps) );
    #else
        cudaMemcpyToSymbol ( props, _props, sizeof(SimProps) );
    #endif
}
//----------------------------------------------------------------------------------------------------------------------
void fillIntZero(cudaStream_t _stream, int _threadsPerBlock, int *_bufferPtr,int size)
{
    if(size>_threadsPerBlock)
    {
        //calculate how many blocks we want
        int blocks = ceil(size/_threadsPerBlock)+1;
        fillIntZeroKernal<<<blocks,_threadsPerBlock,0,_stream>>>(_bufferPtr,size);
    }
    else{
        fillIntZeroKernal<<<1,size,0,_stream>>>(_bufferPtr,size);
    }
    //make sure all our threads are done
    cudaThreadSynchronize();
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Fill int zero: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void createHashMap(cudaStream_t _stream, int _threadsPerBlock, int _hashTableSize, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _hashTableSize;
    if(_hashTableSize>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_hashTableSize/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    // Create ou hash map
    createHashMapKernal<<<blocks,threads,0,_stream>>>(_hashTableSize,_buff);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Create hash map error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void hashAndSort(cudaStream_t _stream,int _threadsPerBlock, int _numParticles, int _hashTableSize, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    //Hash our partilces
    hashParticles<<<blocks,threads,0,_stream>>>(_numParticles,_buff.hashKeys,_buff.cellOccBuffer,_buff.posPtr);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Hash Particles error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    //make sure all our threads are done
    cudaThreadSynchronize();

//    //Turn our raw pointers into thrust pointers so we can use
//    //thrusts sort algorithm
    thrust::device_ptr<int> t_hashPtr = thrust::device_pointer_cast(_buff.hashKeys);
    thrust::device_ptr<float3> t_posPtr = thrust::device_pointer_cast(_buff.posPtr);
    thrust::device_ptr<float3> t_velPtr = thrust::device_pointer_cast(_buff.velPtr);
    thrust::device_ptr<float3> t_accPtr = thrust::device_pointer_cast(_buff.accPtr);
    thrust::device_ptr<int> t_cellOccPtr = thrust::device_pointer_cast(_buff.cellOccBuffer);
    thrust::device_ptr<int> t_cellIdxPtr = thrust::device_pointer_cast(_buff.cellIndexBuffer);

    //sort our buffers
    thrust::sort_by_key(t_hashPtr,t_hashPtr+_numParticles, thrust::make_zip_iterator(thrust::make_tuple(t_posPtr,t_velPtr,t_accPtr)));
    //make sure all our threads are done
    cudaThreadSynchronize();


    //Create our cell indexs
    //run an excludive scan on our arrays to do this
    thrust::exclusive_scan(t_cellOccPtr,t_cellOccPtr+_hashTableSize,t_cellIdxPtr);

    //make sure all our threads are done
    cudaThreadSynchronize();

    //DEBUG: uncomment to print out counted cell occupancy
    //thrust::copy(t_cellOccPtr, t_cellOccPtr+_hashTableSize, std::ostream_iterator<unsigned int>(std::cout, " "));
    //std::cout<<"\n"<<std::endl;
}

//----------------------------------------------------------------------------------------------------------------------
void initDensity(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    //Solve our particles
    solveDensityKernal<<<blocks,threads,0,_stream>>>(_numParticles,_buff);

    //make sure all our threads are done
    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("initDensity error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
}
//----------------------------------------------------------------------------------------------------------------------
void solve(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, float _restDensity, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }


    //Solve for our new positions
    solveForcesKernal<<<blocks,threads,0,_stream>>>(_numParticles, _restDensity, _buff);

    //make sure all our threads are done
    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("Solve error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

}
//----------------------------------------------------------------------------------------------------------------------
bool isConverged(int _numParticles, float &_percentConverged, fluidBuffers _buff)
{
    // Turn our density buffer pointer into a thrust iterater
    thrust::device_ptr<int> t_conPtr = thrust::device_pointer_cast(_buff.convergedPtr);

    // Use reduce to sum all our densities
    float sum = (float)thrust::reduce(t_conPtr, t_conPtr+_numParticles, 0, thrust::plus<int>());

    _percentConverged = (float)sum/(float)_numParticles;

    return (sum>=(float)_numParticles*0.9f/**0.85f*//*0.9962f*/);
}
//----------------------------------------------------------------------------------------------------------------------
bool nanCheck(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    int* isNan = 0;
    int zero = 0;
    checkCudaErrors(cudaMalloc(&isNan,sizeof(int)));
    checkCudaErrors(cudaMemcpy(isNan,&zero,sizeof(int),cudaMemcpyHostToDevice));

    nanCheckKernel<<<blocks,threads,0,_stream>>>(_numParticles,_buff.posPtr,isNan);

    //make sure all our threads are done
    cudaThreadSynchronize();

    checkCudaErrors(cudaMemcpy(&zero,isNan,sizeof(int),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(isNan));
    return zero;
}
//----------------------------------------------------------------------------------------------------------------------
std::vector<float3> sortAndCountValidParticles(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    // compute our PDfs
    std::vector<float> pdf = computeSamplePDFs(_stream,_threadsPerBlock,_numParticles,_buff);

    int *validBuff;
    float3 *outbuffer;
    float *pdfBuff;
    checkCudaErrors(cudaMalloc(&validBuff,sizeof(int)*_numParticles));
    checkCudaErrors(cudaMalloc(&outbuffer,sizeof(float3)*_numParticles));
    checkCudaErrors(cudaMalloc(&pdfBuff,sizeof(float)*_numParticles));
    checkCudaErrors(cudaMemcpy(pdfBuff,&pdf[0],sizeof(float)*_numParticles,cudaMemcpyHostToDevice));


    isValidPos<<<blocks,threads,0,_stream>>>(_numParticles,_buff.convergedPtr,_buff.posPtr,outbuffer,validBuff);

    cudaThreadSynchronize();

    //sort our buffers
    thrust::device_ptr<float3> t_posPtr = thrust::device_pointer_cast(outbuffer);
    thrust::device_ptr<float> t_pdfPtr = thrust::device_pointer_cast(pdfBuff);
    thrust::device_ptr<int> t_validPtr = thrust::device_pointer_cast(validBuff);
    thrust::sort_by_key(t_validPtr,t_validPtr+_numParticles, thrust::make_zip_iterator(thrust::make_tuple(t_posPtr,t_pdfPtr)));

    // set our pos z coordinate to represent that samples pdf
    mergePosAndPDF<<<blocks,threads,0,_stream>>>(_numParticles,outbuffer,pdfBuff);

    cudaThreadSynchronize();

    //count how many are valid
    int numValid = thrust::reduce(t_validPtr, t_validPtr+_numParticles, 0, thrust::plus<int>());

    std::vector<float3> validP;
    validP.resize(numValid);

    checkCudaErrors(cudaMemcpy(&validP[0],outbuffer+(_numParticles-numValid),sizeof(float3)*numValid,cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(validBuff));
    checkCudaErrors(cudaFree(outbuffer));

    return validP;

}
//----------------------------------------------------------------------------------------------------------------------
struct max_functor
{
  __host__ __device__
  float operator()(const float &a, const float &b) const {
    return (a<b)?b:a;
  }
};
//----------------------------------------------------------------------------------------------------------------------
std::vector<float> computeSamplePDFs(cudaStream_t _stream, int _threadsPerBlock, int _numParticles, fluidBuffers _buff)
{
    int blocks = 1;
    int threads = _numParticles;
    if(_numParticles>_threadsPerBlock)
    {
        //calculate how many blocks we want
        blocks = ceil(_numParticles/_threadsPerBlock)+1;
        threads = _threadsPerBlock;
    }

    float *pdfBuff;
    checkCudaErrors(cudaMalloc(&pdfBuff,sizeof(float)*_numParticles));

    // Compute the density of each particle without size function
    solveDensityNoScaleKernal<<<blocks,threads,0,_stream>>>(_numParticles,pdfBuff,_buff);

    cudaThreadSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("solveDensityNoScaleKernal error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }

    // Find the maximum value in our array
    thrust::device_ptr<float> t_pdfBuff = thrust::device_pointer_cast(pdfBuff);

    float sum = thrust::reduce(t_pdfBuff, t_pdfBuff+_numParticles, 0, thrust::plus<float>());


    float maxVal = thrust::reduce(t_pdfBuff, t_pdfBuff+_numParticles, 0, max_functor());

    // Divide our values by our maximum to create vaules between 0-1
    arrayMultiply<<<blocks,threads,0,_stream>>>(_numParticles,pdfBuff,1.f/maxVal);

    // Copy our result to our output array
    std::vector<float> out;
    out.resize(_numParticles);
    checkCudaErrors(cudaMemcpy(&out[0],pdfBuff,sizeof(float)*_numParticles,cudaMemcpyDeviceToHost));
    // Free our allocated device memory
    checkCudaErrors(cudaFree(pdfBuff));

    return out;

}
//----------------------------------------------------------------------------------------------------------------------
