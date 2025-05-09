#include <iostream>
#include <QPainter>
#include <QtMath>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QLabel>
#include <QGraphicsTextItem>

#include "sineanimation.h"
#include "fft_magnitude.h"

SineAnimation::SineAnimation(QWidget *parent)
    : QWidget(parent), phase(0.0), sineAmplitude(0.50), sineFrequency(2.0),
      sawtoothAmplitude(0.50), sawtoothFrequency(2.0) {

    // Scene and View
    scene = new QGraphicsScene(this);
    graphicsView = new QGraphicsView(scene);
    graphicsView->setRenderHint(QPainter::Antialiasing);
    graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // Second Scene and View
    scene2 = new QGraphicsScene(this);
    graphicsView2 = new QGraphicsView(scene2);
    graphicsView2->setRenderHint(QPainter::Antialiasing);
    graphicsView2->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    graphicsView2->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // Sliders
    sineAmplitudeSlider = new QSlider(Qt::Horizontal);
    sineAmplitudeSlider->setRange(0, 100);
    sineAmplitudeSlider->setValue(sineAmplitude * 100);

    sineFrequencySlider = new QSlider(Qt::Horizontal);
    sineFrequencySlider->setRange(1, graphicsView->width());
    sineFrequencySlider->setValue(sineFrequency / 2); // scale up for int slider

    // Labels
    sineAmplitudeLabel = new QLabel(QString("Amplitude: %1").arg(sineAmplitude, 0, 'f', 2));
    sineFrequencyLabel = new QLabel(QString("Frequency: %1").arg(sineFrequency));

    // Sawtooth Sliders
    sawtoothAmplitudeSlider = new QSlider(Qt::Horizontal);
    sawtoothAmplitudeSlider->setRange(0, 100);
    sawtoothAmplitudeSlider->setValue(sawtoothAmplitude * 100);

    sawtoothFrequencySlider = new QSlider(Qt::Horizontal);
    sawtoothFrequencySlider->setRange(1, graphicsView->width());
    sawtoothFrequencySlider->setValue(sawtoothFrequency / 2); // scale up for int slider

    // Sawtooth Labels
    sawtoothAmplitudeLabel = new QLabel(QString("Sawtooth Amplitude: %1").arg(sawtoothAmplitude, 0, 'f', 2));
    sawtoothFrequencyLabel = new QLabel(QString("Sawtooth Frequency: %1").arg(sawtoothFrequency));

    // Layout for canvases
    QVBoxLayout *canvasLayout = new QVBoxLayout();
    canvasLayout->addWidget(graphicsView);
    canvasLayout->addWidget(graphicsView2);

    // Layout for sliders
    QVBoxLayout *sliderLayout = new QVBoxLayout();
    sliderLayout->addWidget(sineAmplitudeLabel);
    sliderLayout->addWidget(sineAmplitudeSlider);
    sliderLayout->addWidget(sineFrequencyLabel);
    sliderLayout->addWidget(sineFrequencySlider);
    sliderLayout->addWidget(sawtoothAmplitudeLabel);
    sliderLayout->addWidget(sawtoothAmplitudeSlider);
    sliderLayout->addWidget(sawtoothFrequencyLabel);
    sliderLayout->addWidget(sawtoothFrequencySlider);

    // Main layout
    QHBoxLayout *mainLayout = new QHBoxLayout(this);
    mainLayout->addLayout(canvasLayout);
    mainLayout->addLayout(sliderLayout);

    // Timer for animation
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &SineAnimation::updateAnimation);
    timer->start(30); // ~30 FPS

    // Connect sliders
    connect(sineAmplitudeSlider, &QSlider::valueChanged, this, &SineAnimation::updateSineAmplitude);
    connect(sineFrequencySlider, &QSlider::valueChanged, this, &SineAnimation::updateSineFrequency);

    // Connect sawtooth sliders
    connect(sawtoothAmplitudeSlider, &QSlider::valueChanged, this, &SineAnimation::updateSawtoothAmplitude);
    connect(sawtoothFrequencySlider, &QSlider::valueChanged, this, &SineAnimation::updateSawtoothFrequency);

    // Initial draw
    drawSineWave();
}

void SineAnimation::drawSineWave() {
    const qreal PI = 3.14159265358979323846;
    const int width = graphicsView->width();
    const int height = graphicsView->height();
    const qreal yCenter = height / 2.0;
    
    sineFrequencySlider->setRange(1, graphicsView->width() / 2);
    sawtoothFrequencySlider->setRange(1, graphicsView->width() / 2);

    path = QPainterPath();
    
    float signal[width];
    float fft_magnitude[width / 2 + 1];

    for (int x = 0; x < width; ++x) {
        float t = static_cast<float>(x) / width;
        float sine_t = sineAmplitude * std::sin(2 * PI * sineFrequency * (t + phase));
        float sawtooth_t;
        sawtooth_t = (t + phase) *  sawtoothFrequency;
        sawtooth_t = abs(sawtooth_t - round(sawtooth_t)) - 0.25;
        sawtooth_t = sawtooth_t * sawtoothAmplitude * 4;
        signal[x] = sine_t + sawtooth_t;
    }
    cudaError_t cudaStatus = compute_fft_magnitude(signal, fft_magnitude, width);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "compute_fft_magnitude failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    for (int x = 0; x < width; ++x) {
        qreal y = yCenter - height * 0.25 * signal[x];
        if (x == 0) {
            path.moveTo(x, y);
        } else {
            path.lineTo(x, y);
        }
    }

    scene->clear();
    scene->addPath(path, QPen(Qt::blue, 2));
    scene->setSceneRect(0, 0, width, height);

    // Add ticks and labels for time plot
    int numberOfTicks = 10;
    for (int i = 0; i < numberOfTicks; ++i) {
        qreal x = static_cast<qreal>(i) / numberOfTicks * width;
        qreal y = height;

        // Tick
        scene->addLine(x, y - 5, x, y, QPen(Qt::black));

        // Label
        qreal tickValue = static_cast<qreal>(i) / numberOfTicks;
        QGraphicsTextItem *text = scene->addText(QString::number(tickValue, 'f', 1));
        text->setPos(x - text->boundingRect().width() / 2, y - text->boundingRect().height());
    }

    const int height2 = graphicsView2->height();

    path = QPainterPath();
    for (int x = 0; x < width / 2 + 1; ++x) {
        qreal y = height2 * 0.95 - height2 * 0.9 * log10(fft_magnitude[x] + 1.0) / 5;
        if (x == 0) {
            path.moveTo(2 * x, y);
        } else {
            path.lineTo(2 * x, y);
        }
    }

    scene2->clear();
    scene2->addPath(path, QPen(Qt::blue, 2));
    scene2->setSceneRect(0, 0, width, height);

    // Add ticks and labels for frequency plot
    numberOfTicks = 5;
    for (int i = 0; i < numberOfTicks; ++i) {
        qreal x = static_cast<qreal>(i) / numberOfTicks * width;
        qreal y = height;

        // Tick
        scene2->addLine(x, y - 5, x, y, QPen(Qt::black));

        // Label
        qreal tickValue = static_cast<qreal>(i) / numberOfTicks / 2;
        QGraphicsTextItem *text = scene2->addText(QString::number(tickValue, 'f', 1));
        text->setPos(x - text->boundingRect().width() / 2, y - text->boundingRect().height());
    }
}

void SineAnimation::updateAnimation() {
    phase += 1.0 / 240;
    drawSineWave();
}

void SineAnimation::updateSineAmplitude(int value) {
    sineAmplitude = static_cast<qreal>(value) / 100;
    sineAmplitudeLabel->setText(QString("Sine Amplitude: %1").arg(sineAmplitude, 0, 'f', 2));
    drawSineWave();
}

void SineAnimation::updateSineFrequency(int value) {
    sineFrequency = static_cast<qreal>(value);
    sineFrequencyLabel->setText(QString("Sine Frequency: %1").arg(sineFrequency));
    drawSineWave();
}

void SineAnimation::updateSawtoothAmplitude(int value) {
    sawtoothAmplitude = static_cast<qreal>(value) / 100;
    sawtoothAmplitudeLabel->setText(QString("Sawtooth Amplitude: %1").arg(sawtoothAmplitude, 0, 'f', 2));
    drawSineWave();
}

void SineAnimation::updateSawtoothFrequency(int value) {
    sawtoothFrequency = static_cast<qreal>(value);
    sawtoothFrequencyLabel->setText(QString("Sawtooth Frequency: %1").arg(sawtoothFrequency));
    drawSineWave();
}