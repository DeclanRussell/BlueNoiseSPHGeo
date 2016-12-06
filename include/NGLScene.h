#ifndef NGLSCENE_H__
#define NGLSCENE_H__

#ifdef DARWIN
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
    #ifndef WIN32
        #include <GL/gl.h>
    #endif
#endif


#include <QGLWidget>
#include <QString>
#include <QTime>

#include <glm/vec3.hpp>
#include "ParticleDrawer.h"
#include "Text.h"
#include "Camera.h"
#include "ShaderProgram.h"
#include "SPHSolverCUDA.h"

//----------------------------------------------------------------------------------------------------------------------
/// @file NGLScene.h
/// @brief this class inherits from the Qt OpenGLWindow and allows us to use NGL to draw OpenGL
/// @author Jonathan Macey
/// @version 1.0
/// @date 10/9/13
/// Revision History :
/// This is an initial version used for the new NGL6 / Qt 5 demos
/// @class NGLScene
/// @brief our main glwindow widget for NGL applications all drawing elements are
/// put in this file
//----------------------------------------------------------------------------------------------------------------------

class NGLScene : public QGLWidget
{
  Q_OBJECT
  public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief ctor for our NGL drawing class
    /// @param [in] parent the parent window to the class
    //----------------------------------------------------------------------------------------------------------------------
    explicit NGLScene(const QGLFormat _format, QWidget *_parent=0);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief dtor must close down ngl and release OpenGL resources
    //----------------------------------------------------------------------------------------------------------------------
    ~NGLScene();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the initialize class is called once when the window is created and we have a valid GL context
    /// use this to setup any default GL stuff
    //----------------------------------------------------------------------------------------------------------------------
    void initializeGL();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief timer to update our fluid simulation
    //----------------------------------------------------------------------------------------------------------------------
    void timerEvent(QTimerEvent *);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this is called everytime we want to draw the scene
    //----------------------------------------------------------------------------------------------------------------------
    void paintGL();
    //----------------------------------------------------------------------------------------------------------------------

    // Qt 5.5.1 must have this implemented and uses it
    void resizeGL(QResizeEvent *_event);
    // Qt 5.x uses this instead! http://doc.qt.io/qt-5/qopenglwindow.html#resizeGL
    void resizeGL(int _w, int _h);

    public slots:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to export pair wise differentials of particles to file
    //----------------------------------------------------------------------------------------------------------------------
    void exportDiffToFile(QString _dir);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to toggle our automatic update
    //----------------------------------------------------------------------------------------------------------------------
    inline void tglUpdate(){m_update = !m_update; if(m_update) {m_startTime = m_startTime.currentTime();}}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to reset our simulation back to white noise
    //----------------------------------------------------------------------------------------------------------------------
    void resetSim();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief slot to set the desired density difference of our simulation. This in turn gives us the ability to create
    /// @brief different noise profiles
    /// @param _diff - desired density difference (double)
    //----------------------------------------------------------------------------------------------------------------------
    inline void setDensDiff(double _diff){m_SPHSolverCUDA->setDensityDiff((float)_diff);}
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Gets a sample from our simulation
    //----------------------------------------------------------------------------------------------------------------------
    inline void getSample(float &_x, float &_y, float &_z){float3 s = m_SPHSolverCUDA->getSample(); _x=s.x;_y=s.y;_z=s.z;}
    //----------------------------------------------------------------------------------------------------------------------
signals:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief updates value of progess bar
    /// @param _x - value to set progress
    //----------------------------------------------------------------------------------------------------------------------
    void updateProgress(int _x);
    //----------------------------------------------------------------------------------------------------------------------
private:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief start time of sim
    //----------------------------------------------------------------------------------------------------------------------
    QTime m_startTime;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief time taken to converge
    //----------------------------------------------------------------------------------------------------------------------
    float m_convergeTime;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief toggele update boolean
    //----------------------------------------------------------------------------------------------------------------------
    bool m_update;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our SPHSolver CUDA version
    //----------------------------------------------------------------------------------------------------------------------
    SPHSolverCUDA *m_SPHSolverCUDA;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our particle drawer
    //----------------------------------------------------------------------------------------------------------------------
    ParticleDrawer *m_particleDrawer;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief used to store the x rotation mouse value
    //----------------------------------------------------------------------------------------------------------------------
    int m_spinXFace;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief used to store the y rotation mouse value
    //----------------------------------------------------------------------------------------------------------------------
    int m_spinYFace;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief flag to indicate if the mouse button is pressed when dragging
    //----------------------------------------------------------------------------------------------------------------------
    bool m_rotate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief flag to indicate if the Right mouse button is pressed when dragging
    //----------------------------------------------------------------------------------------------------------------------
    bool m_translate;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the previous x mouse value
    //----------------------------------------------------------------------------------------------------------------------
    int m_origX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the previous y mouse value
    //----------------------------------------------------------------------------------------------------------------------
    int m_origY;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the previous x mouse value for Position changes
    //----------------------------------------------------------------------------------------------------------------------
    int m_origXPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the previous y mouse value for Position changes
    //----------------------------------------------------------------------------------------------------------------------
    int m_origYPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief window width
    //----------------------------------------------------------------------------------------------------------------------
    int m_width;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief window height
    //----------------------------------------------------------------------------------------------------------------------
    int m_height;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our text drawer
    //----------------------------------------------------------------------------------------------------------------------
    Text *m_text;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief used to store the global mouse transforms
    //----------------------------------------------------------------------------------------------------------------------
    glm::mat4 m_mouseGlobalTX;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Our Camera
    //----------------------------------------------------------------------------------------------------------------------
    Camera m_cam;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief the model position for mouse movement
    //----------------------------------------------------------------------------------------------------------------------
    glm::vec3 m_modelPos;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief M Matrix handle in our shader
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_MLoc;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief MV Matrix handle in our shader
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_MVLoc;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief MVP Matrix handle in our shader
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_MVPLoc;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Normal Matrix handle in our shader
    //----------------------------------------------------------------------------------------------------------------------
    GLuint m_normalMatLoc;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief our phong shader
    //----------------------------------------------------------------------------------------------------------------------
    ShaderProgram *m_phongShader;
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief method to load transform matrices to the shader
    //----------------------------------------------------------------------------------------------------------------------
    void loadMatricesToShader();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Qt Event called when a key is pressed
    /// @param [in] _event the Qt event to query for size etc
    //----------------------------------------------------------------------------------------------------------------------
    void keyPressEvent(QKeyEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called every time a mouse is moved
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mouseMoveEvent (QMouseEvent * _event );
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse button is pressed
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mousePressEvent ( QMouseEvent *_event);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse button is released
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void mouseReleaseEvent ( QMouseEvent *_event );

    //----------------------------------------------------------------------------------------------------------------------
    /// @brief this method is called everytime the mouse wheel is moved
    /// inherited from QObject and overridden here.
    /// @param _event the Qt Event structure
    //----------------------------------------------------------------------------------------------------------------------
    void wheelEvent( QWheelEvent *_event);


};



#endif
