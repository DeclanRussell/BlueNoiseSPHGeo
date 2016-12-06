#include "MainWindow.h"
#include <QFileDialog>
#include <QPushButton>
#include <QDesktopServices>
#include <QGLFormat>
#include <QGroupBox>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent){
    QGroupBox *gb = new QGroupBox(this);
    setCentralWidget(gb);
    m_gridLayout = new QGridLayout(gb);
    gb->setLayout(m_gridLayout);

    QGLFormat format;
    format.setVersion(4,1);
    format.setProfile(QGLFormat::CoreProfile);

    //add our openGL context to our scene
    m_openGLWidget = new NGLScene(format,this);
    m_gridLayout->addWidget(m_openGLWidget,0,0,4,4);
    m_openGLWidget->show();

    // Group box to hold our general controls
    QGroupBox *genGb = new QGroupBox("General Controls",gb);
    m_gridLayout->addWidget(genGb,4,0,1,4);
    // Grid layout for this groupbox
    QGridLayout *genGbLyt = new QGridLayout(genGb);
    genGb->setLayout(genGbLyt);

    // Add some controls to interface with the application
    // Toggle play button
    QPushButton *tglPlayBtn = new QPushButton("Play/Pause",genGb);
    genGbLyt->addWidget(tglPlayBtn,1,0,1,1);
    connect(tglPlayBtn,SIGNAL(pressed()),m_openGLWidget,SLOT(tglUpdate()));

    // Reset button
    QPushButton *rstBtn = new QPushButton("Reset",genGb);
    genGbLyt->addWidget(rstBtn,2,0,1,1);
    connect(rstBtn,SIGNAL(pressed()),m_openGLWidget,SLOT(resetSim()));

    // Write differentials to file button
    QPushButton *diffBtn = new QPushButton("Save differentials",genGb);
    genGbLyt->addWidget(diffBtn,3,0,1,1);
    connect(diffBtn,SIGNAL(pressed()),this,SLOT(saveDiff()));


    // Add our progress bar for stuff
    m_progressBar = new QProgressBar(genGb);
    m_progressBar->setMaximum(100);
    m_progressBar->setMinimum(0);
    connect(m_openGLWidget,SIGNAL(updateProgress(int)),m_progressBar,SLOT(setValue(int)));
    genGbLyt->addWidget(m_progressBar,4,0,1,1);

    // Group box to hold our simulation paramiters
    QGroupBox *prmGb = new QGroupBox("Simualtion Paramiters",gb);
    m_gridLayout->addWidget(prmGb,0,4,5,1);
    // Grid layout for this groupbox
    QGridLayout *prmGbLyt = new QGridLayout(prmGb);
    prmGb->setLayout(prmGbLyt);

    // Density difference field
    prmGbLyt->addWidget(new QLabel("Density Difference:",prmGb),0,0,1,1);
    // Spin box for the value
    QDoubleSpinBox *ddSpnBx = new QDoubleSpinBox(prmGb);
    ddSpnBx->setValue(10.0);
    ddSpnBx->setMaximum(INFINITY);
    ddSpnBx->setMinimum(0);
    ddSpnBx->setDecimals(4);
    ddSpnBx->setSingleStep(10);
    connect(ddSpnBx,SIGNAL(valueChanged(double)),m_openGLWidget,SLOT(setDensDiff(double)));
    prmGbLyt->addWidget(ddSpnBx,0,1,1,1);




    // Gensample field
    prmGbLyt->addWidget(new QLabel("Generate sample:",prmGb),4,0,1,1);
    QPushButton *genSampleBtn = new QPushButton("Generate",genGb);
    prmGbLyt->addWidget(genSampleBtn,4,1,1,1);
    connect(genSampleBtn,SIGNAL(pressed()),this,SLOT(genSample()));
    // Spin box for the value
    m_sampleX = new QDoubleSpinBox(prmGb);
    m_sampleX->setValue(0);
    m_sampleX->setMinimum((double)INFINITY*-1.0);
    m_sampleY = new QDoubleSpinBox(prmGb);
    m_sampleY->setValue(0);
    m_sampleY->setMinimum((double)INFINITY*-1.0);
    m_sampleZ = new QDoubleSpinBox(prmGb);
    m_sampleZ->setValue(1);
    m_sampleZ->setMinimum((double)INFINITY*-1.0);
    prmGbLyt->addWidget(m_sampleX,2,2,1,1);
    prmGbLyt->addWidget(m_sampleY,2,3,1,1);
    prmGbLyt->addWidget(m_sampleZ,2,4,1,1);



}


MainWindow::~MainWindow(){
    //delete m_openGLWidget;
}

void MainWindow::saveDiff()
{
    QString dir = QFileDialog::getSaveFileName(this,"Save Differentials to File");
    if(!dir.isEmpty())
    {
        m_openGLWidget->exportDiffToFile(dir);
    }
}


void MainWindow::genSample()
{
    float x,y,z;
    m_openGLWidget->getSample(x,y,z);
    m_sampleX->setValue(x);
    m_sampleY->setValue(y);
    m_sampleZ->setValue(z);
}
