#ifndef SPHSOLVERCUDA_H
#define SPHSOLVERCUDA_H

//----------------------------------------------------------------------------------------------------------------------
/// @file SPHSolver.h
/// @brief Calculates and updates our new particle positions with navier-stokes equations using CUDA acceleration.
/// @author Declan Russell
/// @version 1.0
/// @date 03/02/2015
/// @class SPHSolverCUDA
//----------------------------------------------------------------------------------------------------------------------

#ifdef OPENGL_BUFFERS
    #ifdef DARWIN
        #include <OpenGL/gl3.h>
    #else
        #include <GL/glew.h>
        #ifndef WIN32
            #include <GL/gl.h>
        #endif
    #endif
#endif

#include <glm/glm.hpp>

// Just another stupid quirk by windows -.-
// Without windows.h defined before cuda_gl_interop you get
// redefinition conflicts.
#ifdef WIN32
    #include <Windows.h>
#endif
#include <cuda_runtime.h>
#include <helper_cuda.h>

#ifdef OPENGL_BUFFERS
#include <cuda_gl_interop.h>
#endif

#include "SPHSolverCUDAKernals.h"
#include <vector>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <QImage>


class SPHSolverCUDA
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our defualt constructor
    /// @param _x - the x boundary of our simulation
    /// @param _y - the y boundary of our simulation
    /// @param _t - the thickness of our boundary
    /// @param _l - the number of layers we want in our boundary
    //----------------------------------------------------------------------------------------------------------------------
    SPHSolverCUDA(float _x = 20.f, float _y = 20.f, float _t = 0.05, float _l = 2);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~SPHSolverCUDA();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Sets the particles init positions in our simulation from an array. If set more than once old data will be removed.
    //----------------------------------------------------------------------------------------------------------------------
    void setParticles(std::vector<float3> &_particles);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Retrieves our particles from the GPU and returns them in a vector
    /// @return array of particle positions (vector<float3>)
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<float3> getParticlePositions();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Retrieves all valid particles for samples. I.e. particles above the hemisphere
    /// @returns array of particle positions (vector<float3>)
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<float3> getValidParticlePositions();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Retrieves our particles from the GPU and returns them in a vector in glm form
    /// @return array of particle positions (vector<float3>)
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<glm::vec3> getParticlePositionsGLM();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Returns a sample in world space from our simulation
    //----------------------------------------------------------------------------------------------------------------------
    float3 getSample();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Returns a sample in world space from our simulation. This version of our function lets you dictate your own rng
    /// @param _rng - random number to select sample index. must be between [0,1]
    //----------------------------------------------------------------------------------------------------------------------
    float3 getSample(float _rng);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Generates _n random positions for our simulation.
    /// @brief Note that these samples will replace the original samples in the simulation.
    /// @param _n - number of samples to generate.
    //----------------------------------------------------------------------------------------------------------------------
    void genRandomSamples(unsigned int _n);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to the number of particles in our simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline int getNumParticles(){return m_simProperties.numParticles;}
    //----------------------------------------------------------------------------------------------------------------------
#ifdef OPENGL_BUFFERS
    /// @brief Returns our OpenGL VAO handle to our particle positions
    /// @return OpenGL VAO handle to our particle positions (GLuint)
    //----------------------------------------------------------------------------------------------------------------------
    inline GLuint getPositionsVAO(){return m_posVAO;}
#endif
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief set the mass of our particles
    /// @param _m - mass of our particles (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline void setMass(float _m){m_simProperties.mass = _m; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to the mass of our particles
    //----------------------------------------------------------------------------------------------------------------------
    inline float getMass(){return m_simProperties.mass;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator for the timestep of our simulation
    /// @param _t - desired timestep
    //----------------------------------------------------------------------------------------------------------------------
    inline void setTimeStep(float _t){m_simProperties.timeStep = _t; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to k our gas/stiffness constant
    /// @param _k - desired gas/stiffness constant
    //----------------------------------------------------------------------------------------------------------------------
    inline void setKConst(float _k){m_simProperties.k = _k; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to k our gas/stiffness constant
    /// @return k our gas/stiffness constant (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getKConst(){return m_simProperties.k;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our smoothing length h
    /// @param _h - desired smoothing length
    //----------------------------------------------------------------------------------------------------------------------
    inline void setSmoothingLength(float _h);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our rest/target density
    /// @param _d - desired rest/target density
    //----------------------------------------------------------------------------------------------------------------------
    inline void setRestDensity(float _d){m_restDensity = _d; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief updates the sim properties on our GPU
    //----------------------------------------------------------------------------------------------------------------------
    inline void updateGPUSimProps(){updateSimProps(&m_simProperties);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our update function to increment the step of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    void update();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief updates our simulation until it has converged
    //----------------------------------------------------------------------------------------------------------------------
    void updateUntilConverged();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief compute the average density of our simulation
    /// @return average density of simulation (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getAverageDensity(){return computeAverageDensity(m_simProperties.numParticles,m_fluidBuffers);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our density difference
    /// @param _diff - desired density difference
    //----------------------------------------------------------------------------------------------------------------------
    inline void setDensityDiff(float _diff){m_densityDiff = _diff;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief mutator to our convergence value
    /// @param _x - desired convergence value (int)
    //----------------------------------------------------------------------------------------------------------------------
    inline void setConvergeValue(float _x){m_simProperties.convergeValue = _x; updateGPUSimProps();}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief accessor to our convergence value
    /// @return convergence value (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getConvergeValue(){return m_simProperties.convergeValue;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief checks to see if our simulation has convered
    /// @return is our simulation has convereged (bool)
    //----------------------------------------------------------------------------------------------------------------------
    inline bool convergedState(float &_percentConverged){return isConverged(m_simProperties.numParticles,_percentConverged,m_fluidBuffers);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief a debugging function to print out our particle positions
    //----------------------------------------------------------------------------------------------------------------------
    void printParticlePos();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief shuffles our valid samples buffer
    //----------------------------------------------------------------------------------------------------------------------
    void shuffleValidSamples();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Use an image to set the light intensity to generate our samples for
    /// @param _img - image to represent light intensity
    //----------------------------------------------------------------------------------------------------------------------
    void setLightIntensity(QImage _img);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief computes PDF for some intensity
    /// @param _i - intensity (float)
    //----------------------------------------------------------------------------------------------------------------------
    inline float getPDF(float _i){return _i*m_simProperties.sfWidth.y+m_simProperties.sfWidth.x;}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief computes sample pdfs
    /// @returns _array of sample pdfs (std::vector<float>)
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<float> getSamplePDFs();
    //----------------------------------------------------------------------------------------------------------------------
protected:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief function to set our hash grid position and dimensions
    /// @param _gridMin - minimum position of our grid
    /// @param _gridDim - grid dimentions
    //----------------------------------------------------------------------------------------------------------------------
    void setHashPosAndDim(float3 _gridMin, float3 _gridDim);
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief size function bandwidth
    //----------------------------------------------------------------------------------------------------------------------
    float m_sfWidth;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our maximum threads per block.
    //----------------------------------------------------------------------------------------------------------------------
    int m_threadsPerBlock;
    //----------------------------------------------------------------------------------------------------------------------
#ifdef OPENGL_BUFFERS
    /// @brief VAO handle of our positions buffer
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_posVAO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief VBO handle to our positions buffer
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_posVBO;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our cuda graphics resource for our particle positions OpenGL interop.
    //----------------------------------------------------------------------------------------------------------------------
    cudaGraphicsResource_t m_resourcePos;
#endif
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our fluid buffers on our device
    //----------------------------------------------------------------------------------------------------------------------
    fluidBuffers m_fluidBuffers;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Structure to hold all of our simulation properties so we can easily pass it to our CUDA kernal.
    //----------------------------------------------------------------------------------------------------------------------
    SimProps m_simProperties;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Rest/Target density of our fluid
    //----------------------------------------------------------------------------------------------------------------------
    float m_restDensity;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our CUDA stream to help run kernals concurrently
    //----------------------------------------------------------------------------------------------------------------------
    cudaStream_t m_cudaStream;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our 2nd CUDA stream to help run kernals concurrently
    //----------------------------------------------------------------------------------------------------------------------
    cudaStream_t m_cudaStream2;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the density difference of our simulation
    //----------------------------------------------------------------------------------------------------------------------
    float m_densityDiff;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our rng generator
    //----------------------------------------------------------------------------------------------------------------------
    boost::variate_generator< boost::mt19937, boost::uniform_real<float> > *m_gen;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Vector to store our generated particle positions
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<float3> m_hostPartPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Boolean if our host stored particles positions are different to that of the device
    //----------------------------------------------------------------------------------------------------------------------
    bool m_hostPosDirty;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the number of samples that we have taken
    //----------------------------------------------------------------------------------------------------------------------
    int m_numSamplesTaken;
    //----------------------------------------------------------------------------------------------------------------------
};

#endif // SPHSOLVERKERNALS_H

