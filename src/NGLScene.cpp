#include "NGLScene.h"

#include <QMouseEvent>
#include <QGuiApplication>
#include <fstream>
#include <vector>
#include <string>
#include <fstream>

#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "SPHSolverCUDA.h"

#include <openvdb/openvdb.h>

#define DtoR M_PI/180.0f

//----------------------------------------------------------------------------------------------------------------------
/// @brief the increment for x/y translation with mouse movement
//----------------------------------------------------------------------------------------------------------------------
const static float INCREMENT=0.01f;
//----------------------------------------------------------------------------------------------------------------------
/// @brief the increment for the wheel zoom
//----------------------------------------------------------------------------------------------------------------------
const static float ZOOM=0.5f;
//----------------------------------------------------------------------------------------------------------------------
NGLScene::NGLScene(const QGLFormat _format, QWidget *_parent) : QGLWidget(_format,_parent)
{
  // set this widget to have the initial keyboard focus
  setFocus();
  setFocusPolicy( Qt::StrongFocus );
  // re-size the widget to that of the parent (in that case the GLFrame passed in on construction)
  m_rotate=false;
  // Set update to false so we dont update our simulation as soon as the app launches
  m_update = false;
  // mouse rotation values set to 0
  m_spinXFace=0.0f;
  m_spinYFace=0.0f;
}
//----------------------------------------------------------------------------------------------------------------------
NGLScene::~NGLScene()
{
  if(m_model) delete m_model;
  m_model = 0;
  delete m_SPHSolverCUDA;
  delete m_particleDrawer;
}
//----------------------------------------------------------------------------------------------------------------------
void NGLScene::resizeGL(QResizeEvent *_event)
{
  m_width=_event->size().width()*devicePixelRatio();
  m_height=_event->size().height()*devicePixelRatio();
  m_text->setScreenSize(m_width,m_height);
  // now set the camera size values as the screen size has changed
  m_cam.setShape(45.0f,(float)width(),(float)height(),0.05f,350.0f);
  m_particleDrawer->setScreenWidth(_event->size().width());
}
//----------------------------------------------------------------------------------------------------------------------
void NGLScene::resizeGL(int _w , int _h)
{
  m_cam.setShape(45.0f,(float)_w,(float)_h,0.05f,350.0f);
  m_width=_w*devicePixelRatio();
  m_height=_h*devicePixelRatio();
  m_text->setScreenSize(m_width,m_height);
  m_particleDrawer->setScreenWidth(_w);
}
//----------------------------------------------------------------------------------------------------------------------
void NGLScene::exportDiffToFile(QString _dir)
{
    std::vector<float3> particles = m_SPHSolverCUDA->getParticlePositions();
    float3 p1,p2;
    float2 p1v2, p2v2;
    float2 d;
    float rLength;
    float progress;
    std::fstream diffs(_dir.toStdString().c_str(),std::fstream::out | std::fstream::trunc);
    if(!diffs.is_open())
    {
        std::cout<<"Opening file "<<_dir.toStdString()<<" has failed!"<<std::endl;
        return;
    }
    int totalDiffs = ((int)particles.size()*(int)particles.size())-(int)particles.size();
    for(unsigned int i=0;i<particles.size();i++)
    {
        p1 = particles[i];
        for(unsigned int j=0;j<particles.size();j++)
        {
            if(i==j) continue;
            p2 = particles[j];

            // Check to see if they are on the same hemisphere if not then lets just skip
            if(!(p1.y*p2.y>0.f)) continue;

            // Get the arclength
            rLength = (20.f)*asin(length(p1-p2)/(20.f));

            // Turn themm into 2d coords as we only need the direction and distance
            p1v2 = make_float2(p1.x,p1.y);
            p2v2 = make_float2(p2.x,p2.y);

            // Normalise the corrdinates
            d = p2v2 - p1v2;
            d/= length(d);

            // Scale our differencial by our arc legnth
            d*=rLength;
            diffs<<d.x<<" "<<d.y<<std::endl;

            progress = ((float)(j+(i*particles.size()))/(float)(totalDiffs))*100.f;
            updateProgress((int) progress);
        }
    }
    diffs.close();
}
//----------------------------------------------------------------------------------------------------------------------
void NGLScene::resetSim()
{
    //m_SPHSolver->setParticles(testParticles);
    m_SPHSolverCUDA->genRandomSamples(m_SPHSolverCUDA->getNumParticles());
    //m_particleDrawer->setPositions(positions);
}
//----------------------------------------------------------------------------------------------------------------------
void NGLScene::importModel(QString _loc)
{
    delete m_model;
    m_model = new Model(_loc.toStdString());
}
//----------------------------------------------------------------------------------------------------------------------
void NGLScene::initializeGL()
{
#ifndef DARWIN
    glewExperimental = GL_TRUE;
    GLenum error = glewInit();
    if(error != GLEW_OK){
        std::cerr<<"GLEW IS NOT OK!!! "<<std::endl;
    }
#endif

  glClearColor(1.f, 1.f, 1.f, 1.0f);			   // White Background
  // enable depth testing for drawing
  glEnable(GL_DEPTH_TEST);
  // enable multisampling for smoother drawing
  glEnable(GL_MULTISAMPLE);

  // Now we will create a basic Camera from the graphics library
  // This is a static camera so it only needs to be set once
  // First create Values for the camera position
  glm::vec3 from(0,0,25);
  glm::vec3 to(0,0,0);
  glm::vec3 up(0,1,0);
  // now load to our new camera
  m_cam = Camera(from,to,up);
  // set the shape using FOV 45 Aspect Ratio based on Width and Height
  // The final two are near and far clipping planes of 0.5 and 10
  m_cam.setShape(45.0f,720.0f,576.0f,0.05f,350.0f);

  // Create our phong shader program
  m_phongShader = new ShaderProgram();
  // now we are going to create our shaders from source
  Shader vert("shaders/PhongVertex.glsl",GL_VERTEX_SHADER);
  Shader frag("shaders/PhongFragment.glsl",GL_FRAGMENT_SHADER);
  // attach the shaders to program
  m_phongShader->attachShader(&vert);
  m_phongShader->attachShader(&frag);
  m_phongShader->bindFragDataLocation(0, "fragColour");
  // now we have associated that data we can link the shader
  m_phongShader->link();
  // and make it active ready to
  m_phongShader->use();

  //Set some uniforms for our shader
  glUniform3f(m_phongShader->getUniformLoc("viewerPos"),from.x,from.y,from.z);
  glUniform3f(m_phongShader->getUniformLoc("material.ambient"),0.274725f,0.1995f,0.0745f);
  glUniform3f(m_phongShader->getUniformLoc("material.diffuse"),0.75164f,0.60648f,0.22648f);
  glUniform3f(m_phongShader->getUniformLoc("material.specular"),0.628281f,0.555802f,0.3666065f);
  glUniform1f(m_phongShader->getUniformLoc("material.shininess"),51.2f);

  glUniform4f(m_phongShader->getUniformLoc("light.position"),0.f,5.f,0.f,1.f);
  glUniform4f(m_phongShader->getUniformLoc("light.diffuse"),1.f,1.f,1.f,1.f);
  glUniform4f(m_phongShader->getUniformLoc("light.ambient"),1.f,1.f,1.f,1.f);
  glUniform4f(m_phongShader->getUniformLoc("light.specular"),1.f,1.f,1.f,1.f);
  glUniform1f(m_phongShader->getUniformLoc("light.constantAttenuation"),1.f);
  glUniform1f(m_phongShader->getUniformLoc("light.quadraticAttenuation"),0.f);
  glUniform1f(m_phongShader->getUniformLoc("light.linearAttenuation"),0.f);
  glUniform1f(m_phongShader->getUniformLoc("light.spotCosCutoff"),180.f);

  m_MLoc = m_phongShader->getUniformLoc("M");
  m_MVLoc = m_phongShader->getUniformLoc("MV");
  m_MVPLoc = m_phongShader->getUniformLoc("MVP");
  m_normalMatLoc = m_phongShader->getUniformLoc("normalMatrix");


  //Create our text drawer
  m_text = new Text(QFont("Ariel"));
  m_text->setScreenSize(width(),height());
  m_text->setColour(1.f,0.f,0.f);

  //Create our nice efficient particle drawer
  m_particleDrawer = new ParticleDrawer;
  m_particleDrawer->setParticleSize(0.2f);
  m_particleDrawer->setScreenWidth(width());
  m_particleDrawer->setColour(1.f,0.f,0.f);

  //Create our SPH Solver
  m_SPHSolverCUDA = new SPHSolverCUDA;
  m_SPHSolverCUDA->genRandomSamples(22000);

  m_model = new Model("models/newteapot.obj");
  std::cout<<"Num faces: "<<m_model->getNumFaces()<<std::endl;
  std::cout<<"Model has "<<m_model->getNumVerts()<<" verts"<<std::endl;

  // initialize openvdb
  openvdb::initialize();



  // Start our timer event. This will begin calling the TimerEvent function that updates our simulation.
  startTimer(0);
}


void NGLScene::loadMatricesToShader()
{
  m_phongShader->use();
  glm::mat4 MV;
  glm::mat4 MVP;
  glm::mat3 normalMatrix;
  glm::mat4 M = m_mouseGlobalTX;
  //M = m_mouseGlobalTX*M;
  MV = m_cam.getViewMatrix()*M;
  MVP= m_cam.getProjectionMatrix()*m_cam.getViewMatrix()*M;
  normalMatrix=glm::mat3(MV);
  normalMatrix = glm::inverse(normalMatrix);
  normalMatrix = glm::transpose(normalMatrix);
  glUniformMatrix4fv(m_MVLoc, 1, GL_FALSE,glm::value_ptr(MV));
  glUniformMatrix4fv(m_MVPLoc, 1, GL_FALSE,glm::value_ptr(MVP));
  glUniformMatrix3fv(m_normalMatLoc, 1, GL_FALSE,glm::value_ptr(normalMatrix));
  glUniformMatrix4fv(m_MLoc, 1, GL_FALSE,glm::value_ptr(M));

}

//----------------------------------------------------------------------------------------------------------------------
void NGLScene::timerEvent(QTimerEvent *)
{
    if(m_update)
    {
        m_SPHSolverCUDA->update();
    }
    updateGL();
}
//----------------------------------------------------------------------------------------------------------------------
void NGLScene::paintGL()
{
  glViewport(0,0,m_width,m_height);
  // clear the screen and depth buffer
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  m_phongShader->use();

  // Rotation based on the mouse position for our global transform
  glm::mat4 rotX;
  glm::mat4 rotY;
  // create the rotation matrices
  rotX = glm::rotate(rotX,(float)m_spinXFace*(float)DtoR,glm::vec3(1,0,0));
  rotY = glm::rotate(rotY,(float)m_spinYFace*(float)DtoR,glm::vec3(0,1,0));
  // multiply the rotations
  m_mouseGlobalTX=rotY*rotX;
  // add the translations
  m_mouseGlobalTX[3][0] = m_modelPos.x;
  m_mouseGlobalTX[3][1] = m_modelPos.y;
  m_mouseGlobalTX[3][2] = m_modelPos.z;
  loadMatricesToShader();

  //draw our VAO
  glBindVertexArray(m_model->getVAO());
  glDrawArrays(GL_TRIANGLES,0,m_model->getNumVerts());

  m_particleDrawer->setColour(1.f,0.f,0.f);  //red
#ifdef OPENGL_BUFFERS
  //m_particleDrawer->drawFromVAO(m_SPHSolverCUDA->getPositionsVAO(),m_SPHSolverCUDA->getNumParticles(), m_mouseGlobalTX,m_cam.getViewMatrix(),m_cam.getProjectionMatrix());
#else
  m_particleDrawer->setPositions(m_SPHSolverCUDA->getParticlePositionsGLM());
  m_particleDrawer->draw(m_mouseGlobalTX,m_cam.getViewMatrix(),m_cam.getProjectionMatrix());
#endif

  QTime currentTime;
  currentTime = currentTime.currentTime();


  QString text;
  float pc;
  if(m_SPHSolverCUDA->convergedState(pc))
  {
      if(m_update)
      {
        m_convergeTime = m_startTime.msecsTo(currentTime) / 1000.f;
      }
      m_update = false;
      text = QString("Simulation has converged! Epsilon value: %1 NumParticles: %2 Time taken to converge: %3").arg(m_SPHSolverCUDA->getConvergeValue()).arg(m_SPHSolverCUDA->getNumParticles()).arg(m_convergeTime);
      m_text->setColour(0.f,1.f,0.f);
  }
  else
  {
      if(m_update)
      {
        m_convergeTime = m_startTime.msecsTo(currentTime) / 1000.f;
        text = QString("Simulation converging. Epsilon value: %1 NumParticles: %2 Time taken to converge: %3 Percent Converged %4").arg(m_SPHSolverCUDA->getConvergeValue()).arg(m_SPHSolverCUDA->getNumParticles()).arg(m_convergeTime).arg(pc);
        m_text->setColour(1.f,0.f,0.f);
      }
      else
      {
        text = QString("Simulation paused. Epsilon value: %1 NumParticles: %2 Time taken to converge: %3 Percent Converged %4").arg(m_SPHSolverCUDA->getConvergeValue()).arg(m_SPHSolverCUDA->getNumParticles()).arg(m_convergeTime).arg(pc);
        m_text->setColour(0.f,1.f,1.f);
      }
  }
  m_text->renderText(0,0,text);


}
//----------------------------------------------------------------------------------------------------------------------
void NGLScene::mouseMoveEvent (QMouseEvent * _event)
{
  // note the method buttons() is the button state when event was called
  // that is different from button() which is used to check which button was
  // pressed when the mousePress/Release event is generated
  if(m_rotate && _event->buttons() == Qt::LeftButton)
  {
    int diffx=_event->x()-m_origX;
    int diffy=_event->y()-m_origY;
    m_spinXFace += (float) 0.5f * diffy;
    m_spinYFace += (float) 0.5f * diffx;
    m_origX = _event->x();
    m_origY = _event->y();
    update();

  }
        // right mouse translate code
  else if(m_translate && _event->buttons() == Qt::RightButton)
  {
    int diffX = (int)(_event->x() - m_origXPos);
    int diffY = (int)(_event->y() - m_origYPos);
    m_origXPos=_event->x();
    m_origYPos=_event->y();
    m_modelPos.x += INCREMENT * diffX;
    m_modelPos.y -= INCREMENT * diffY;
    update();

   }
}


//----------------------------------------------------------------------------------------------------------------------
void NGLScene::mousePressEvent ( QMouseEvent * _event)
{
  // that method is called when the mouse button is pressed in this case we
  // store the value where the maouse was clicked (x,y) and set the Rotate flag to true
  if(_event->button() == Qt::LeftButton)
  {
    m_origX = _event->x();
    m_origY = _event->y();
    m_rotate =true;
  }
  // right mouse translate mode
  else if(_event->button() == Qt::RightButton)
  {
    m_origXPos = _event->x();
    m_origYPos = _event->y();
    m_translate=true;
  }

}

//----------------------------------------------------------------------------------------------------------------------
void NGLScene::mouseReleaseEvent ( QMouseEvent * _event )
{
  // that event is called when the mouse button is released
  // we then set Rotate to false
  if (_event->button() == Qt::LeftButton)
  {
    m_rotate=false;
  }
        // right mouse translate mode
  if (_event->button() == Qt::RightButton)
  {
    m_translate=false;
  }
}

//----------------------------------------------------------------------------------------------------------------------
void NGLScene::wheelEvent(QWheelEvent *_event)
{

	// check the diff of the wheel position (0 means no change)
	if(_event->delta() > 0)
	{
        m_modelPos.z+=ZOOM;
	}
	else if(_event->delta() <0 )
	{
        m_modelPos.z-=ZOOM;
	}
	update();
}
//----------------------------------------------------------------------------------------------------------------------

void NGLScene::keyPressEvent(QKeyEvent *_event)
{
  // that method is called every time the main window recives a key event.
  // we then switch on the key value and set the camera in the GLWindow
  switch (_event->key())
  {
  // escape key to quit
  case Qt::Key_Escape : QGuiApplication::exit(EXIT_SUCCESS); break;
  // turn on wirframe rendering
  case Qt::Key_W : glPolygonMode(GL_FRONT_AND_BACK,GL_LINE); break;
  // turn off wire frame
  case Qt::Key_S : glPolygonMode(GL_FRONT_AND_BACK,GL_FILL); break;
  // show full screen
  case Qt::Key_F : showFullScreen(); break;
  // show windowed
  case Qt::Key_N : showNormal(); break;
  // update simulation by one step
  case Qt::Key_E : m_SPHSolverCUDA->update(); break;
  // reset our simulation
  case Qt::Key_R : resetSim(); break;
  // toggle update automatically
  case Qt::Key_Space : m_update = !m_update; break;
      // Change size of particles drawn
  case Qt::Key_Minus : m_particleDrawer->setParticleSize(m_particleDrawer->getParticleSize()-.05f); break;
  case Qt::Key_Plus : m_particleDrawer->setParticleSize(m_particleDrawer->getParticleSize()+.05f); break;
  case Qt::Key_P : m_SPHSolverCUDA->printParticlePos(); break;
  case Qt::Key_C : m_SPHSolverCUDA->updateUntilConverged(); break;
  default : break;
  }
  // finally update the GLWindow and re-draw
  //if (isExposed())
    update();
}
