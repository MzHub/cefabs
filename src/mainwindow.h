//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2011 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
#pragma once

#include "ui_mainwindow.h"
#include "imageview.h"
#include "videoplayer.h"
#include "cpu_image.h"

class MainWindow : public QMainWindow, protected Ui_MainWindow, public ImageView::Handler {
    Q_OBJECT
public:
    MainWindow();
    ~MainWindow();

    void restoreSettings();
    void closeEvent(QCloseEvent *e);

    const QImage& image() const { return m_result[m_select->currentIndex()]; }

protected slots:
    void on_actionOpen_triggered();
    void on_actionAbout_triggered();
    void on_actionSelectDevice_triggered();
    void on_actionRecord_triggered();

    void setDirty();
    void process();
    
    void onIndexChanged(int);
    void onVideoChanged(int nframes);

signals:
    void imageChanged(const QImage&);

protected:
    virtual void draw(ImageView *view, QPainter &p, const QRectF& R, const QImage& image);

    VideoPlayer *m_player;
    cpu_image<float4> m_st;
    cpu_image<float4> m_tfm;
    QImage m_result[2];
    bool m_dirty;

    int m_N;
    double m_sigma_d;
    double m_tau_r;
    double m_sigma_t;
    double m_max_angle;
    double m_sigma_i;
    double m_sigma_g;
    double m_r;
    double m_tau_s;
    double m_sigma_a;
};
