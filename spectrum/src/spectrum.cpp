#include <QApplication>
#include "sineanimation.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    SineAnimation window;
    window.resize(1500, 800);
    window.setWindowTitle("Spectrum");
    window.show();

    return app.exec();
}