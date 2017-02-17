/******************************************************************************
 *
 * $Id$
 *
 * Project:  WindNinja Qt GUI
 * Purpose:  Handles surface inputs for the domain
 * Author:   Kyle Shannon <ksshannon@gmail.com>
 *
 ******************************************************************************
 *
 * THIS SOFTWARE WAS DEVELOPED AT THE ROCKY MOUNTAIN RESEARCH STATION (RMRS)
 * MISSOULA FIRE SCIENCES LABORATORY BY EMPLOYEES OF THE FEDERAL GOVERNMENT 
 * IN THE COURSE OF THEIR OFFICIAL DUTIES. PURSUANT TO TITLE 17 SECTION 105 
 * OF THE UNITED STATES CODE, THIS SOFTWARE IS NOT SUBJECT TO COPYRIGHT 
 * PROTECTION AND IS IN THE PUBLIC DOMAIN. RMRS MISSOULA FIRE SCIENCES 
 * LABORATORY ASSUMES NO RESPONSIBILITY WHATSOEVER FOR ITS USE BY OTHER 
 * PARTIES,  AND MAKES NO GUARANTEES, EXPRESSED OR IMPLIED, ABOUT ITS QUALITY, 
 * RELIABILITY, OR ANY OTHER CHARACTERISTIC.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef SURFACEINPUT_H_
#define SURFACEINPUT_H_

#include <QGroupBox>
#include <QLineEdit>
#include <QToolButton>
#include <QComboBox>
#include <QSpinBox>
#include <QFileDialog>
#include <QLabel>
#include <QRadioButton>
#include <QObject>
#include <QMessageBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>

#include "WindNinjaInputs.h"
#include "timeZoneWidget.h"
#include "WidgetDownloadDEM.h"

class surfaceInput : public QWidget
{
  Q_OBJECT

public:
    surfaceInput(QWidget *parent = 0);

#ifdef NINJAFOAM
    QGroupBox *foamCaseGroupBox;
    QLineEdit *foamCaseLineEdit;
    QToolButton *foamCaseOpenToolButton;
#endif

    QGroupBox *inputFileGroupBox;

    QLineEdit *inputFileLineEdit;
    QToolButton *inputFileOpenToolButton;
    QToolButton *downloadDEMButton;

    QGroupBox *roughnessGroupBox;
    QComboBox *roughnessComboBox;
    QLabel *roughnessLabel;

    QGroupBox *meshResGroupBox;
    QComboBox *meshResComboBox;
    QDoubleSpinBox *meshResDoubleSpinBox;
    QRadioButton *meshMetersRadioButton, *meshFeetRadioButton;

    QGroupBox *timeZoneGroupBox;
    timeZoneWidget *timeZone;

#ifdef NINJAFOAM
    QHBoxLayout *foamCaseLayout;
#endif
    QHBoxLayout *inputLayout;
    QHBoxLayout *roughnessLayout;
    QHBoxLayout *roughnessLabelLayout;
    QHBoxLayout *meshResLayout;
    QHBoxLayout *timeZoneLayout;
    QVBoxLayout *mainLayout;
    
signals:
    void writeToMainConsole(QString message);

};

#endif /* SURFACEINPUT_H_ */

