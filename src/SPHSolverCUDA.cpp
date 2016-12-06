#include "SPHSolverCUDA.h"
#include <iostream>
#define SpeedOfSound 34.29f
#include <helper_math.h>
#include <algorithm>
#include <ctime>
#include <QColor>


//----------------------------------------------------------------------------------------------------------------------
SPHSolverCUDA::SPHSolverCUDA(float _x, float _y, float _t, float _l)
{
    //Lets test some cuda stuff
    int count;
    if (cudaGetDeviceCount(&count))
        return;
    std::cout << "Found " << count << " CUDA device(s)" << std::endl;
    if(count == 0)
    {
        std::cerr<<"Install an Nvidia chip!"<<std::endl;
        exit(-1);
    }
    for (int i=0; i < count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout<<prop.name<<", Compute capability:"<<prop.major<<"."<<prop.minor<<std::endl;;
        std::cout<<"  Global mem: "<<prop.totalGlobalMem/ 1024 / 1024<<"M, Shared mem per block: "<<prop.sharedMemPerBlock / 1024<<"k, Registers per block: "<<prop.regsPerBlock<<std::endl;
        std::cout<<"  Warp size: "<<prop.warpSize<<" threads, Max threads per block: "<<prop.maxThreadsPerBlock<<", Multiprocessor count: "<<prop.multiProcessorCount<<" MaxBlocks: "<<prop.maxGridSize[0]<<std::endl;
        m_threadsPerBlock = prop.maxThreadsPerBlock;
    }

    m_hostPosDirty = true;
    m_numSamplesTaken = 0;

    // Create our CUDA stream to run our kernals on. This helps with running kernals concurrently.
    // Check them out at http://on-demand.gputechconf.com/gtc-express/2011/presentations/StreamsAndConcurrencyWebinar.pdf
    checkCudaErrors(cudaStreamCreate(&m_cudaStream));
    checkCudaErrors(cudaStreamCreate(&m_cudaStream2));

    // Make sure these are init to 0
    m_fluidBuffers.posPtr = 0;
    m_fluidBuffers.accPtr = 0;
    m_fluidBuffers.newPosPtr = 0;
    m_fluidBuffers.velPtr = 0;
    m_fluidBuffers.cellIndexBuffer = 0;
    m_fluidBuffers.cellOccBuffer = 0;
    m_fluidBuffers.hashKeys = 0;
    m_fluidBuffers.hashMap = 0;
    m_fluidBuffers.denPtr = 0;
    m_fluidBuffers.convergedPtr = 0;
    m_fluidBuffers.imgPtr = 0;

    m_sfWidth = 8;
    m_simProperties.sfWidth.x = 1.f/m_sfWidth;
    m_simProperties.sfWidth.y = 1.f-m_simProperties.sfWidth.x;
    m_simProperties.simBounds = make_float3(_x,_y,0.f);
    m_simProperties.gridDim = make_float3(0.f,0.f,0.f);
    m_simProperties.timeStep = 0.004f;
    m_simProperties.gravity = make_float3(0.f,-9.8f,0.f);
    m_simProperties.convergeValue = 0.001f;//0.009f;
    m_simProperties.k = SpeedOfSound*40.f;//SpeedOfSound;
    m_simProperties.mass = 0.05f;
    m_simProperties.accLimit = 200.f;
    m_simProperties.accLimit2 = m_simProperties.accLimit*m_simProperties.accLimit;
    m_simProperties.velLimit = 70.f;
    m_simProperties.velLimit2 = m_simProperties.velLimit*m_simProperties.velLimit;
    m_restDensity = 100.f;
    m_densityDiff = 10.f;
    float _h = .3f;
    m_simProperties.h = _h;
    m_simProperties.hSqrd = _h*_h;
    m_simProperties.dWConst = 315.f/(64.f*(float)M_PI*_h*_h*_h*_h*_h*_h*_h*_h*_h);
    m_simProperties.pWConst = -45.f/((float)M_PI*_h*_h*_h*_h*_h*_h);
    m_simProperties.vWConst = 45.f/((float)M_PI*_h*_h*_h*_h*_h*_h);

    // Send these to the GPU
    updateGPUSimProps();

#ifdef OPENGL_BUFFERS
    // Create an OpenGL buffer for our position buffer
    // Create our VAO and vertex buffers
    glGenVertexArrays(1, &m_posVAO);
    glBindVertexArray(m_posVAO);

    // Put our vertices into an OpenGL buffer
    glGenBuffers(1, &m_posVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    // We must alocate some space otherwise cuda cannot register it
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3), NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourcePos, m_posVBO, cudaGraphicsRegisterFlagsWriteDiscard));

    // Unbind everything just in case
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

#endif

    //Define our boundaries only temperary until we have mesh data
    float3 hmin = make_float3(-_t,-_t,0.f);
    float3 hmax = make_float3(m_simProperties.simBounds.x+m_simProperties.h+_t,m_simProperties.simBounds.y+m_simProperties.h+_t,1.f);
    setHashPosAndDim(hmin,hmax-hmin);

    // setup our rng
    boost::mt19937 rng(time(NULL));
    boost::uniform_real<float> MinusPlusOneFloatDistrib(0.f, 1.f);
    m_gen = new boost::variate_generator< boost::mt19937, boost::uniform_real<float> >(rng, MinusPlusOneFloatDistrib);

}
//----------------------------------------------------------------------------------------------------------------------
SPHSolverCUDA::~SPHSolverCUDA()
{
#ifdef OPENGL_BUFFERS
    // Make sure we remember to unregister our cuda resource
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourcePos));
#endif
    // Delete our CUDA buffers
    if(m_fluidBuffers.posPtr) checkCudaErrors(cudaFree(m_fluidBuffers.posPtr));
    if(m_fluidBuffers.newPosPtr) checkCudaErrors(cudaFree(m_fluidBuffers.newPosPtr));
    if(m_fluidBuffers.velPtr) checkCudaErrors(cudaFree(m_fluidBuffers.velPtr));
    if(m_fluidBuffers.accPtr) checkCudaErrors(cudaFree(m_fluidBuffers.accPtr));
    if(m_fluidBuffers.cellIndexBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellIndexBuffer));
    if(m_fluidBuffers.cellOccBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellOccBuffer));
    if(m_fluidBuffers.hashKeys) checkCudaErrors(cudaFree(m_fluidBuffers.hashKeys));
    if(m_fluidBuffers.hashMap) checkCudaErrors(cudaFree(m_fluidBuffers.hashMap));
    if(m_fluidBuffers.denPtr) checkCudaErrors(cudaFree(m_fluidBuffers.denPtr));
    if(m_fluidBuffers.convergedPtr) checkCudaErrors(cudaFree(m_fluidBuffers.convergedPtr));
    if(m_fluidBuffers.imgPtr) checkCudaErrors(cudaFree(m_fluidBuffers.imgPtr));
    // Delete our CUDA streams as well
    checkCudaErrors(cudaStreamDestroy(m_cudaStream));
    checkCudaErrors(cudaStreamDestroy(m_cudaStream2));
#ifdef OPENGL_BUFFERS
    // Delete our openGL objects
    glDeleteBuffers(1,&m_posVBO);
    glDeleteVertexArrays(1,&m_posVAO);
#endif
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setParticles(std::vector<float3> &_particles)
{
    // Set how many particles we have
    m_simProperties.numParticles = (int)_particles.size();

#ifdef OPENGL_BUFFERS
    // Unregister our resource
    checkCudaErrors(cudaGraphicsUnregisterResource(m_resourcePos));

    // Fill our buffer with our positions
    glBindVertexArray(m_posVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*_particles.size(), &_particles[0], GL_DYNAMIC_DRAW);

    // create our cuda graphics resource for our vertexs used for our OpenGL interop
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_resourcePos, m_posVBO, cudaGraphicsRegisterFlagsWriteDiscard));
#else
    if(m_fluidBuffers.posPtr) checkCudaErrors(cudaFree(m_fluidBuffers.posPtr));
    m_fluidBuffers.posPtr = 0;
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.posPtr,_particles.size()*sizeof(float3)));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.posPtr,&_particles[0],sizeof(float3)*_particles.size(),cudaMemcpyHostToDevice));
#endif


    // Delete our CUDA buffers fi they have anything in them
    if(m_fluidBuffers.newPosPtr) checkCudaErrors(cudaFree(m_fluidBuffers.newPosPtr));
    if(m_fluidBuffers.velPtr) checkCudaErrors(cudaFree(m_fluidBuffers.velPtr));
    if(m_fluidBuffers.accPtr) checkCudaErrors(cudaFree(m_fluidBuffers.accPtr));
    if(m_fluidBuffers.denPtr) checkCudaErrors(cudaFree(m_fluidBuffers.denPtr));
    if(m_fluidBuffers.hashKeys) checkCudaErrors(cudaFree(m_fluidBuffers.hashKeys));
    if(m_fluidBuffers.convergedPtr) checkCudaErrors(cudaFree(m_fluidBuffers.convergedPtr));
    m_fluidBuffers.newPosPtr = 0;
    m_fluidBuffers.velPtr = 0;
    m_fluidBuffers.accPtr = 0;
    m_fluidBuffers.denPtr = 0;
    m_fluidBuffers.hashKeys = 0;
    m_fluidBuffers.convergedPtr = 0;

    // Fill them up with some blank data
    std::vector<float3> blankFloat3s;
    blankFloat3s.resize(_particles.size());
    for(unsigned int i=0;i<blankFloat3s.size();i++) blankFloat3s[i] = make_float3(0.f,0.f,0.f);

    // Send the data to the GPU
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.newPosPtr,blankFloat3s.size()*sizeof(float3)));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.newPosPtr,&blankFloat3s[0],sizeof(float3)*blankFloat3s.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.velPtr,blankFloat3s.size()*sizeof(float3)));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.velPtr,&blankFloat3s[0],sizeof(float3)*blankFloat3s.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.accPtr,blankFloat3s.size()*sizeof(float3)));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.accPtr,&blankFloat3s[0],sizeof(float3)*blankFloat3s.size(),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.hashKeys,_particles.size()*sizeof(int)));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.convergedPtr,_particles.size()*sizeof(int)));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.denPtr,_particles.size()*sizeof(float)));
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.hashKeys,(int)_particles.size());
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.convergedPtr,(int)_particles.size());

}
//----------------------------------------------------------------------------------------------------------------------
std::vector<float3> SPHSolverCUDA::getParticlePositions()
{
    if(!m_hostPosDirty) return m_hostPartPos;

    std::vector<float3> positions;
    positions.resize(m_simProperties.numParticles);

#ifdef OPENGL_BUFFERS
    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));

    // Copy our data from the GPU
    checkCudaErrors(cudaMemcpy(&positions[0],m_fluidBuffers.posPtr,sizeof(float3)*m_simProperties.numParticles,cudaMemcpyDeviceToHost));

    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));
    m_fluidBuffers.posPtr = 0;
#else
    // Copy our data from the GPU
    checkCudaErrors(cudaMemcpy(&positions[0],m_fluidBuffers.posPtr,sizeof(float3)*m_simProperties.numParticles,cudaMemcpyDeviceToHost));
#endif




    // Randomly shuffle our vector so that its not sorted in our hash positions
    std::random_shuffle ( positions.begin(), positions.end() );

    m_hostPartPos = positions;

    m_numSamplesTaken = 0;

    m_hostPosDirty = false;

    return m_hostPartPos;
}
//----------------------------------------------------------------------------------------------------------------------
std::vector<float3> SPHSolverCUDA::getValidParticlePositions()
{
    if(!m_hostPosDirty) return m_hostPartPos;

#ifdef OPENGL_BUFFERS
    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));
#endif

    std::vector<float3> positions = sortAndCountValidParticles(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers);

#ifdef OPENGL_BUFFERS
    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));
    m_fluidBuffers.posPtr = 0;
#endif

    // Randomly shuffle our vector so that its not sorted in our hash positions
    std::random_shuffle ( positions.begin(), positions.end() );

    m_hostPartPos = positions;

    m_numSamplesTaken = 0;

    m_hostPosDirty = false;

    return m_hostPartPos;
}
//----------------------------------------------------------------------------------------------------------------------
std::vector<glm::vec3> SPHSolverCUDA::getParticlePositionsGLM()
{
    /// @todo
    ///
    /// This function isnt really needed its just for debugging so need to remove in the plugin
    ///
    ///


    std::vector<glm::vec3> positionsGLM;
    if(!m_hostPosDirty)
    {
        positionsGLM.resize(m_hostPartPos.size());
        for(unsigned int i=0;i<m_hostPartPos.size();i++)
            positionsGLM[i] = glm::vec3(m_hostPartPos[i].x,m_hostPartPos[i].y,m_hostPartPos[i].z);

        return positionsGLM;
    }

    std::vector<float3> positions;
    positions.resize(m_simProperties.numParticles);

#ifdef OPENGL_BUFFERS
    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));

    // Copy our data from the GPU
    checkCudaErrors(cudaMemcpy(&positions[0],m_fluidBuffers.posPtr,sizeof(float3)*m_simProperties.numParticles,cudaMemcpyDeviceToHost));

    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));
    m_fluidBuffers.posPtr = 0;
#else
    // Copy our data from the GPU
    checkCudaErrors(cudaMemcpy(&positions[0],m_fluidBuffers.posPtr,sizeof(float3)*m_simProperties.numParticles,cudaMemcpyDeviceToHost));
#endif


    // Randomly shuffle our vector so that its not sorted in our hash positions
    std::random_shuffle ( positions.begin(), positions.end() );

    m_hostPartPos = positions;

    m_numSamplesTaken = 0;

    positionsGLM.resize(m_hostPartPos.size());

    for(unsigned int i=0;i<m_hostPartPos.size();i++)
        positionsGLM[i] = glm::vec3(m_hostPartPos[i].x,m_hostPartPos[i].y,m_hostPartPos[i].z);

    return positionsGLM;


}
//----------------------------------------------------------------------------------------------------------------------
float3 SPHSolverCUDA::getSample()
{
    if(getNumParticles()==0)
    {
        std::cerr<<"No particles gerated to sample"<<std::endl;
        return make_float3(0.f,0.f,0.f);
    }

    std::vector<float3> p = getValidParticlePositions();
    m_numSamplesTaken++;
    if(m_numSamplesTaken>=p.size())
    {
        m_numSamplesTaken = 0;
    }

    float3 s = p[m_numSamplesTaken];
    s.x/=m_simProperties.simBounds.x;
    s.y/=m_simProperties.simBounds.y;

    return s;
}
//----------------------------------------------------------------------------------------------------------------------
float3 SPHSolverCUDA::getSample(float _rng)
{
    if(getNumParticles()==0)
    {
        std::cerr<<"No particles gerated to sample"<<std::endl;
        return make_float3(0.f,0.f,0.f);
    }

    std::vector<float3> p = getValidParticlePositions();
    unsigned int i = _rng*(p.size()-1);

    float3 s = p[i];
    s.x/=m_simProperties.simBounds.x;
    s.y/=m_simProperties.simBounds.y;

    return s;
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::genRandomSamples(unsigned int _n)
{
    float3 tempVec;
    std::vector<float3> positionsfloat;
    positionsfloat.resize(_n);
    for(unsigned int i=0;i<_n;i++)
    {
        tempVec = make_float3((*m_gen)(),(*m_gen)(),(*m_gen)());
        //tempVec.x*=tempVec.x;
        while(tempVec.x==0.f || tempVec.x==1.f || tempVec.y==0.f || tempVec.y==1.f) tempVec = make_float3((*m_gen)(),(*m_gen)(),(*m_gen)());
        tempVec.x*=m_simProperties.simBounds.x;
        tempVec.y*=m_simProperties.simBounds.y;
        tempVec.z*=m_simProperties.simBounds.z;
        positionsfloat[i] = make_float3(tempVec.x,tempVec.y,tempVec.z);
    }
    setParticles(positionsfloat);
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setSmoothingLength(float _h)
{
    m_simProperties.h = _h;
    m_simProperties.hSqrd = _h*_h;
    m_simProperties.dWConst = 315.f/(64.f*(float)M_PI*_h*_h*_h*_h*_h*_h*_h*_h*_h);
    m_simProperties.pWConst = -45.f/((float)M_PI*_h*_h*_h*_h*_h*_h);
    m_simProperties.vWConst = 45.f/((float)M_PI*_h*_h*_h*_h*_h*_h);

    setHashPosAndDim(m_simProperties.gridMin,m_simProperties.gridDim);
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setHashPosAndDim(float3 _gridMin, float3 _gridDim)
{
    m_simProperties.gridMin = _gridMin;
    m_simProperties.gridDim = _gridDim;
    m_simProperties.gridRes.x = (int)ceil(_gridDim.x/(m_simProperties.h*m_sfWidth));
    m_simProperties.gridRes.y = (int)ceil(_gridDim.y/(m_simProperties.h*m_sfWidth));
    m_simProperties.gridRes.z = (int)ceil(_gridDim.z/(m_simProperties.h*m_sfWidth));
    m_simProperties.gridRes.x = max(m_simProperties.gridRes.x,1);
    m_simProperties.gridRes.y = max(m_simProperties.gridRes.y,1);
    m_simProperties.gridRes.z = max(m_simProperties.gridRes.z,1);
    int tableSize = ceil(m_simProperties.gridRes.x *m_simProperties.gridRes.y *m_simProperties.gridRes.z);

    // No point in alocating a buffer size of zero so lets just return
    if(tableSize==0)return;

    std::cout<<"table size "<<tableSize<<std::endl;

    // Remove anything that is in our bufferes currently
    if(m_fluidBuffers.cellIndexBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellIndexBuffer));
    if(m_fluidBuffers.cellOccBuffer) checkCudaErrors(cudaFree(m_fluidBuffers.cellOccBuffer));
    if(m_fluidBuffers.hashMap) checkCudaErrors(cudaFree(m_fluidBuffers.hashMap));
    m_fluidBuffers.cellIndexBuffer = 0;
    m_fluidBuffers.cellOccBuffer = 0;
    m_fluidBuffers.hashMap = 0;
    // Send the data to our GPU buffers
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.cellIndexBuffer,tableSize*sizeof(int)));
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.cellOccBuffer,tableSize*sizeof(int)));
    // Fill with blank data
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellOccBuffer,tableSize);
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellIndexBuffer,tableSize);

    // Update this our simulation properties on the GPU
    updateGPUSimProps();

    // Allocate memory for our hash map
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.hashMap,tableSize*sizeof(cellInfo)));

    // Compute our hash map
    createHashMap(m_cudaStream,m_threadsPerBlock,tableSize,m_fluidBuffers);

    cudaThreadSynchronize();

}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::update()
{
    //if no particles then theres no point in updating so just return
    if(!m_simProperties.numParticles)return;

    m_hostPosDirty = true;

    // Set our hash table values back to zero
    int tableSize = ceil(m_simProperties.gridRes.x *m_simProperties.gridRes.y *m_simProperties.gridRes.z);
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellIndexBuffer,tableSize);
    fillIntZero(m_cudaStream,m_threadsPerBlock,m_fluidBuffers.cellOccBuffer,tableSize);

    //Send our sim properties to the GPU
    updateSimProps(&m_simProperties);

#ifdef OPENGL_BUFFERS
    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));
#endif

    // Hash and sort our particles
    hashAndSort(m_cudaStream, m_threadsPerBlock, m_simProperties.numParticles, tableSize , m_fluidBuffers);

    if(nanCheck(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers))
    {
        std::cerr<<"There is nan in hash\n\n\n"<<std::endl;
        //printParticlePos();
        for(int i=0;i<100000;i++){}//waste some time
        exit(-1);
    }

    // Compute our density
    initDensity(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers);

    if(nanCheck(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers))
    {
        std::cerr<<"There is nan in density\n\n\n"<<std::endl;
        for(int i=0;i<100000;i++){}//waste some time
        exit(-1);
    }

    // Compute our rest density
    m_restDensity = getAverageDensity() - m_densityDiff;
    //printParticlePos();

    if(nanCheck(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers))
    {
        std::cerr<<"There is nan in density\n\n\n"<<std::endl;
        for(int i=0;i<100000;i++){}//waste some time
        exit(-1);
    }

    checkCudaErrors(cudaMemcpy(m_fluidBuffers.newPosPtr,m_fluidBuffers.posPtr,sizeof(float3)*m_simProperties.numParticles,cudaMemcpyDeviceToDevice));

    // Solve for our new positions
    solve(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_restDensity,m_fluidBuffers);

    checkCudaErrors(cudaMemcpy(m_fluidBuffers.posPtr,m_fluidBuffers.newPosPtr,sizeof(float3)*m_simProperties.numParticles,cudaMemcpyDeviceToDevice));

    if(nanCheck(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers))
    {
        std::cerr<<"There is nan in solve\n\n\n"<<std::endl;
        //printParticlePos();
        for(int i=0;i<100000;i++){}//waste some time
        exit(-1);
    }

#ifdef OPENGL_BUFFERS
    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));
    m_fluidBuffers.posPtr = 0;
#endif
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::updateUntilConverged()
{
    update();
    float pc;
    while(!convergedState(pc))
    {
        update();
    }
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::printParticlePos()
{
    std::vector<float3> p = getValidParticlePositions();
    for(unsigned int i=0;i<p.size();i++)
    {
        std::cout<<"idx "<<i<<" "<<p[i].x<<","<<p[i].y<<" pdf: "<<p[i].z<<std::endl;
    }
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::shuffleValidSamples()
{
    // Randomly shuffle our vector so that its not sorted in our hash positions
    std::random_shuffle ( m_hostPartPos.begin(), m_hostPartPos.end() );

    m_numSamplesTaken = 0;
}
//----------------------------------------------------------------------------------------------------------------------
void SPHSolverCUDA::setLightIntensity(QImage _img)
{
    m_simProperties.imgRes = make_int2(_img.width(),_img.height());
    updateGPUSimProps();

    // Get our information from our image into a 1D buffer
    std::vector<float> intensity;
    intensity.resize(m_simProperties.imgRes.x*m_simProperties.imgRes.y);
    // Image will always be upside down so we load it in reverse in the y
    int i=0;
    QColor c;

    for(int y=_img.height()-1;y>-1;y--)
    for(int x=0;x<_img.width();x++)
    {
        c = QColor(_img.pixel(x,y));
        intensity[i] = (0.2125 * c.redF()) + (0.7154 * c.greenF()) + (0.0721 * c.blueF());
        i++;
    }

    if(m_fluidBuffers.imgPtr) checkCudaErrors(cudaFree(m_fluidBuffers.imgPtr));
    m_fluidBuffers.imgPtr = 0;
    checkCudaErrors(cudaMalloc(&m_fluidBuffers.imgPtr,sizeof(float)*intensity.size()));
    checkCudaErrors(cudaMemcpy(m_fluidBuffers.imgPtr,&intensity[0],sizeof(float)*intensity.size(),cudaMemcpyHostToDevice));

}
//----------------------------------------------------------------------------------------------------------------------
std::vector<float> SPHSolverCUDA::getSamplePDFs()
{
#ifdef OPENGL_BUFFERS
    // Map our pointer for our position data
    size_t posSize;
    checkCudaErrors(cudaGraphicsMapResources(1,&m_resourcePos));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fluidBuffers.posPtr,&posSize,m_resourcePos));
#endif
    std::vector<float> pdf = computeSamplePDFs(m_cudaStream,m_threadsPerBlock,m_simProperties.numParticles,m_fluidBuffers);
#ifdef OPENGL_BUFFERS
    //unmap our buffer pointer and set it free into the wild
    checkCudaErrors(cudaGraphicsUnmapResources(1,&m_resourcePos));
    m_fluidBuffers.posPtr = 0;
#endif

    return pdf;
}
//----------------------------------------------------------------------------------------------------------------------
