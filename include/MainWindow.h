#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QGroupBox>
#include <QGridLayout>
#include <QSpacerItem>
#include <iostream>
#include <QGridLayout>
#include <QProgressBar>
#include <QDoubleSpinBox>

#include "NGLScene.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
public slots:
    // Slot to save our differentials
    void saveDiff();


    // Gererate sample slot
    void genSample();

    // Get Model
    void getModel();

private:
    NGLScene *m_openGLWidget;
    QGridLayout *m_gridLayout;
    QProgressBar *m_progressBar;
    QDoubleSpinBox *m_sampleX;
    QDoubleSpinBox *m_sampleY;
    QDoubleSpinBox *m_sampleZ;




};

#endif // MAINWINDOW_H
