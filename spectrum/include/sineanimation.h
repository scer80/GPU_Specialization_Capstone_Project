#ifndef SINEANIMATION_H
#define SINEANIMATION_H

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QTimer>
#include <QPainterPath>
#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>

class SineAnimation : public QWidget {
    Q_OBJECT

public:
    SineAnimation(QWidget *parent = nullptr);

private slots:
    void updateAnimation();
    void updateSineAmplitude(int value);
    void updateSineFrequency(int value);
    void updateSawtoothAmplitude(int value);
    void updateSawtoothFrequency(int value);

private:
    QGraphicsScene *scene;
    QGraphicsView *graphicsView;
    QGraphicsScene *scene2;
    QGraphicsView *graphicsView2;
    QPainterPath path;
    QTimer *timer;
    qreal phase;
    qreal sineAmplitude;
    qreal sineFrequency;
    qreal sawtoothAmplitude;
    qreal sawtoothFrequency;

    QSlider *sineAmplitudeSlider;
    QSlider *sineFrequencySlider;
    QSlider *sawtoothAmplitudeSlider;
    QSlider *sawtoothFrequencySlider;

    QLabel *sineAmplitudeLabel;
    QLabel *sineFrequencyLabel;
    QLabel *sawtoothAmplitudeLabel;
    QLabel *sawtoothFrequencyLabel;

    void drawSineWave();
};

#endif // SINEANIMATION_H