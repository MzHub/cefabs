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
#include "mainwindow.h"
#include "param.h"
#include "paramui.h"
#include "cudadevicedialog.h"
#include "imageutil.h"
#include "gpu_image.h"
#include "gpu_ivacef.h"
#include "gpu_color.h"
#include "gpu_st.h"
#include "gpu_stgauss2.h"
#include "gpu_gauss.h"
#include "gpu_util.h"


MainWindow::MainWindow() {
    setupUi(this);
    m_dirty = false;

    m_imageView->setFocus();
    m_imageView->setHandler(this);

    new ParamInt   (this, "N", 5, 1, 100, 1, &m_N);
    new ParamDouble(this, "sigma_d", 1.0, 0.0, 10.0, 0.05, &m_sigma_d);
    new ParamDouble(this, "tau_r", 0.002, 0.0, 1.0, 0.001, &m_tau_r);
    new ParamDouble(this, "sigma_t", 6.0, 0.0, 20.0, 1, &m_sigma_t);
    new ParamDouble(this, "max_angle", 22.5, 0.0, 90.0, 1, &m_max_angle);
    new ParamDouble(this, "sigma_i", 0.0, 0.0, 10.0, 0.25, &m_sigma_i);
    new ParamDouble(this, "sigma_g", 1.5, 0.0, 10.0, 0.25, &m_sigma_g);
    new ParamDouble(this, "r", 2, 0.0, 10.0, 0.25, &m_r);
    new ParamDouble(this, "tau_s", 0.005, -2, 2, 0.01, &m_tau_s);
    new ParamDouble(this, "sigma_a", 1.5, 0.0, 10.0, 0.25, &m_sigma_a);

    ParamUI *pui = new ParamUI(this, this);
    pui->setFixedWidth(200);
    pui->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    m_vbox1->addWidget(pui);
    m_vbox1->addStretch(100);

    connect(m_select, SIGNAL(currentIndexChanged(int)), this, SLOT(onIndexChanged(int)));

    m_player = new VideoPlayer(this, ":/test.png");
    connect(m_player, SIGNAL(videoChanged(int)), this, SLOT(onVideoChanged(int)));
    connect(m_player, SIGNAL(currentFrameChanged(int)), this, SLOT(setDirty()));
    connect(m_player, SIGNAL(outputChanged(const QImage&)), m_imageView, SLOT(setImage(const QImage&)));
    connect(this, SIGNAL(imageChanged(const QImage&)), m_player, SLOT(setOutput(const QImage&)));

    m_videoControls->setFrameStyle(QFrame::NoFrame);
    m_videoControls->setAutoHide(true);
    connect(m_videoControls, SIGNAL(stepForward()), m_player, SLOT(stepForward()));
    connect(m_videoControls, SIGNAL(stepBack()), m_player, SLOT(stepBack()));
    connect(m_videoControls, SIGNAL(currentFrameTracked(int)), m_player, SLOT(setCurrentFrame(int)));
    connect(m_videoControls, SIGNAL(playbackChanged(bool)), m_player, SLOT(setPlayback(bool)));
    connect(m_videoControls, SIGNAL(trackingChanged(bool)), this, SLOT(setDirty()));

    connect(m_player, SIGNAL(videoChanged(int)), m_videoControls, SLOT(setFrameCount(int)));
    connect(m_player, SIGNAL(playbackChanged(bool)), m_videoControls, SLOT(setPlayback(bool)));
    connect(m_player, SIGNAL(currentFrameChanged(int)), m_videoControls, SLOT(setCurrentFrame(int)));
}


MainWindow::~MainWindow() {
}


void MainWindow::restoreSettings() {
    QSettings settings;
    restoreGeometry(settings.value("mainWindow/geometry").toByteArray());
    restoreState(settings.value("mainWindow/windowState").toByteArray());

    settings.beginGroup("imageView");
    m_imageView->restoreSettings(settings);
    settings.endGroup();

    settings.beginGroup("parameters");
    AbstractParam::restoreSettings(settings, this);
    settings.endGroup();

    m_player->restoreSettings(settings);
}


void MainWindow::closeEvent(QCloseEvent *e) {
    QSettings settings;
    settings.setValue("mainWindow/geometry", saveGeometry());
    settings.setValue("mainWindow/windowState", saveState());

    settings.beginGroup("imageView");
    m_imageView->saveSettings(settings);
    settings.endGroup();

    settings.beginGroup("parameters");
    AbstractParam::saveSettings(settings, this);
    settings.endGroup();

    m_player->saveSettings(settings);

    QMainWindow::closeEvent(e);
} 


void MainWindow::on_actionOpen_triggered() {
    m_player->open();
}


void MainWindow::on_actionAbout_triggered() {
    QMessageBox msgBox;
    msgBox.setWindowTitle("About");
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText(
        "<html><body>" \
        "<p><b>Coherence-Enhancing Filtering on the GPU</b><br/><br/>" \
        "Copyright (C) 2010-2011 Hasso-Plattner-Institut,<br/>" \
        "Fachgebiet Computergrafische Systeme &lt;" \
        "<a href='http://www.hpi3d.de'>www.hpi3d.de</a>&gt;<br/><br/>" \
        "Author: Jan Eric Kyprianidis &lt;" \
        "<a href='http://www.kyprianidis.com'>www.kyprianidis.com</a>&gt;<br/>" \
        "Date: " __DATE__ "</p>" \
        "<p>This program is free software: you can redistribute it and/or modify " \
        "it under the terms of the GNU General Public License as published by " \
        "the Free Software Foundation, either version 3 of the License, or " \
        "(at your option) any later version.</p>" \
        "<p>This program is distributed in the hope that it will be useful, " \
        "but WITHOUT ANY WARRANTY; without even the implied warranty of " \
        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the " \
        "GNU General Public License for more details.</p>" \
        "Related Publications:" \
        "<ul>" \
        "<li>" \
        "Kyprianidis, J. E., &amp; Kang, H. (2011). " \
        "Image and Video Abstraction by Coherence-Enhancing Filtering. " \
        "<em>Computer Graphics Forum</em>, 30(2), 593-602. " \
        "(Proceedings Eurographics 2011)" \
        "</li>" \
        "</ul>" \
        "<p>Test image courtesy of Ivan Mlinar @ flickr.com.</p>" \
        "</body></html>"
    );
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.exec();
}


void MainWindow::on_actionSelectDevice_triggered() {
    int current = 0;
    cudaGetDevice(&current);
    int N = CudaDeviceDialog::select(true);
    if ((N >= 0) && (current != N)) {
        QMessageBox::information(this, "Information", "Application must be restarted!");
        qApp->quit();
    }
}

void MainWindow::on_actionRecord_triggered() {
    m_player->record();
}


void MainWindow::setDirty() {
    if (m_videoControls->isTracking()) {
        imageChanged(m_player->image());
    }
    else if (!m_dirty) {
        m_dirty = true;
        QMetaObject::invokeMethod(this, "process", Qt::QueuedConnection);
    }
}


void MainWindow::process() {
    m_dirty = false;
    QImage src = m_player->image();
    if (src.isNull()) {
        m_result[0] = m_result[1] = src;
        imageChanged(image());
        return;
    }

    gpu_image<float4> img = gpu_image_from_qimage<float4>(src);
    gpu_image<float4> st;
    gpu_image<float4> tfm;

    for (int k = 0; k < m_N; ++k) {
        st = gpu_ivacef_sobel(img, st, m_sigma_d, m_tau_r);
        img = gpu_stgauss2_filter(img, st, m_sigma_t, m_max_angle, true, true, true, 2, 1);

        st = gpu_ivacef_sobel(img, st, m_sigma_d, m_tau_r);
        tfm = gpu_st_tfm(st);   

        gpu_image<float> L = gpu_rgb2gray(img);
        L =  gpu_gauss_filter_xy(L, m_sigma_i);
        img = gpu_ivacef_shock(L, img, tfm, m_sigma_g, m_tau_s, m_r);
    }

    img = gpu_stgauss2_filter(img, st, m_sigma_a, 90, false, true, true, 2, 1);
    
    m_st = st.cpu();
    m_tfm = tfm.cpu();
    m_result[0] = src;
    m_result[1] = gpu_image_to_qimage(img);

    imageChanged(image());
}


void MainWindow::onIndexChanged(int index) {
    imageChanged(image());
}


void MainWindow::onVideoChanged(int nframes) {
    gpu_cache_clear();
    window()->setWindowFilePath(m_player->filename());
    window()->setWindowTitle(m_player->filename() + "[*]"); 
    actionRecord->setEnabled(nframes > 1);
}


void MainWindow::draw(ImageView *view, QPainter &p, const QRectF& R, const QImage& image) {
    Handler::draw(view, p, R, image);
    if (m_imageView->scale() > 6) {
        QRect aR = R.toAlignedRect();

        p.setPen(QPen(Qt::blue, 1 / m_imageView->scale()));
        for (int j = aR.top(); j <= aR.bottom(); ++j) {
            for (int i = aR.left(); i <= aR.right(); ++i) {
                float4 t = m_tfm(i, j);
                            
                QPointF q(i+0.5, j+0.5);
                QPointF v(0.9 * t.x-0.5f, 0.9 * t.y-0.5f);
                            
                p.drawLine(q-v, q+v);
            }
        }
    }

    if (m_st.is_valid() && (m_imageView->scale() > 6)) {
        QPointF c = R.center();
        std::vector<float3> path = gpu_stgauss2_path( c.x(), c.y(), m_st, m_sigma_t, m_max_angle, true, true, 2, 0.25f);

        QPolygonF P;
        for (int i = 0; i < (int)path.size(); ++i) {
            P.append(QPointF(path[i].x, path[i].y));
        }

        p.setPen(QPen(Qt::black, view->pt2px(2), Qt::SolidLine, Qt::RoundCap));
        p.drawPolyline(P);
        if (m_imageView->scale() > 30) {
            p.setPen(QPen(Qt::black, view->pt2px(5.0), Qt::SolidLine, Qt::RoundCap));
            p.drawPoints(P);
        }
    }
}
