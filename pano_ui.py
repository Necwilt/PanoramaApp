# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pano.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1440, 741)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.resultImg = QtWidgets.QLabel(self.centralwidget)
        self.resultImg.setGeometry(QtCore.QRect(20, 20, 871, 621))
        self.resultImg.setMinimumSize(QtCore.QSize(400, 400))
        self.resultImg.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultImg.setLineWidth(1)
        self.resultImg.setMidLineWidth(1)
        self.resultImg.setText("")
        self.resultImg.setPixmap(QtGui.QPixmap("imageplaceholder.png"))
        self.resultImg.setScaledContents(True)
        self.resultImg.setObjectName("resultImg")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(899, 19, 231, 461))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_2 = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.comboBox = QtWidgets.QComboBox(self.frame_2)
        self.comboBox.setGeometry(QtCore.QRect(10, 20, 137, 22))
        self.comboBox.setIconSize(QtCore.QSize(20, 20))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(10, 0, 161, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setGeometry(QtCore.QRect(10, 60, 211, 21))
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setObjectName("label_2")
        self.match_conf = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.match_conf.setGeometry(QtCore.QRect(10, 90, 141, 31))
        self.match_conf.setSingleStep(0.05)
        self.match_conf.setProperty("value", 0.65)
        self.match_conf.setObjectName("match_conf")
        self.conf_thresh = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.conf_thresh.setGeometry(QtCore.QRect(10, 160, 141, 31))
        self.conf_thresh.setSingleStep(0.05)
        self.conf_thresh.setProperty("value", 1.0)
        self.conf_thresh.setObjectName("conf_thresh")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setGeometry(QtCore.QRect(10, 130, 231, 21))
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setObjectName("label_3")
        self.wave_correct = QtWidgets.QComboBox(self.frame_2)
        self.wave_correct.setGeometry(QtCore.QRect(10, 230, 137, 21))
        self.wave_correct.setIconSize(QtCore.QSize(20, 20))
        self.wave_correct.setObjectName("wave_correct")
        self.wave_correct.addItem("")
        self.wave_correct.addItem("")
        self.wave_correct.addItem("")
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setGeometry(QtCore.QRect(10, 200, 211, 21))
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.frame_2)
        self.label_5.setGeometry(QtCore.QRect(10, 270, 211, 21))
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setObjectName("label_5")
        self.warp = QtWidgets.QComboBox(self.frame_2)
        self.warp.setGeometry(QtCore.QRect(10, 300, 137, 21))
        self.warp.setIconSize(QtCore.QSize(20, 20))
        self.warp.setObjectName("warp")
        self.warp.addItem("")
        self.warp.addItem("")
        self.warp.addItem("")
        self.warp.addItem("")
        self.warp.addItem("")
        self.warp.addItem("")
        self.warp.addItem("")
        self.frame = QtWidgets.QFrame(self.frame_2)
        self.frame.setGeometry(QtCore.QRect(0, 350, 221, 80))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.crop = QtWidgets.QCheckBox(self.frame)
        self.crop.setGeometry(QtCore.QRect(10, 10, 201, 20))
        self.crop.setObjectName("crop")
        self.runButton = QtWidgets.QPushButton(self.frame)
        self.runButton.setGeometry(QtCore.QRect(10, 40, 201, 28))
        self.runButton.setObjectName("runButton")
        self.verticalLayout_3.addWidget(self.frame_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1440, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.editMode = QtWidgets.QMenu(self.menubar)
        self.editMode.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.editMode.setObjectName("editMode")
        self.debugMode = QtWidgets.QMenu(self.menubar)
        self.debugMode.setObjectName("debugMode")
        self.analyticalMode = QtWidgets.QMenu(self.menubar)
        self.analyticalMode.setObjectName("analyticalMode")
        self.presets = QtWidgets.QMenu(self.menubar)
        self.presets.setObjectName("presets")
        self.manualMode = QtWidgets.QMenu(self.menubar)
        self.manualMode.setObjectName("manualMode")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget.setMinimumSize(QtCore.QSize(300, 300))
        self.dockWidget.setAcceptDrops(True)
        self.dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetFloatable|QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)
        self.analyticDock = QtWidgets.QDockWidget(MainWindow)
        self.analyticDock.setFloating(False)
        self.analyticDock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.analyticDock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.analyticDock.setObjectName("analyticDock")
        self.dockWidgetContents_2 = QtWidgets.QWidget()
        self.dockWidgetContents_2.setMinimumSize(QtCore.QSize(1, 1))
        self.dockWidgetContents_2.setObjectName("dockWidgetContents_2")
        self.gridLayoutWidget = QtWidgets.QWidget(self.dockWidgetContents_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(130, 30, 160, 89))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.dock_layout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.dock_layout.setContentsMargins(0, 0, 0, 0)
        self.dock_layout.setSpacing(0)
        self.dock_layout.setObjectName("dock_layout")
        self.analyticOutput = QtWidgets.QTextEdit(self.gridLayoutWidget)
        self.analyticOutput.setReadOnly(True)
        self.analyticOutput.setObjectName("analyticOutput")
        self.dock_layout.addWidget(self.analyticOutput, 0, 0, 1, 1)
        self.analyticDock.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.analyticDock)
        self.loadImages = QtWidgets.QAction(MainWindow)
        self.loadImages.setObjectName("loadImages")
        self.clearImages = QtWidgets.QAction(MainWindow)
        self.clearImages.setObjectName("clearImages")
        self.saveImg = QtWidgets.QAction(MainWindow)
        self.saveImg.setObjectName("saveImg")
        self.exit = QtWidgets.QAction(MainWindow)
        self.exit.setObjectName("exit")
        self.editEnable = QtWidgets.QAction(MainWindow)
        self.editEnable.setCheckable(True)
        self.editEnable.setObjectName("editEnable")
        self.editDisable = QtWidgets.QAction(MainWindow)
        self.editDisable.setCheckable(False)
        self.editDisable.setObjectName("editDisable")
        self.debugEnable = QtWidgets.QAction(MainWindow)
        self.debugEnable.setCheckable(True)
        self.debugEnable.setObjectName("debugEnable")
        self.analyticEnable = QtWidgets.QAction(MainWindow)
        self.analyticEnable.setCheckable(True)
        self.analyticEnable.setObjectName("analyticEnable")
        self.preset1 = QtWidgets.QAction(MainWindow)
        self.preset1.setObjectName("preset1")
        self.preset3 = QtWidgets.QAction(MainWindow)
        self.preset3.setObjectName("preset3")
        self.preset4 = QtWidgets.QAction(MainWindow)
        self.preset4.setObjectName("preset4")
        self.preset5 = QtWidgets.QAction(MainWindow)
        self.preset5.setObjectName("preset5")
        self.presetDefault = QtWidgets.QAction(MainWindow)
        self.presetDefault.setObjectName("presetDefault")
        self.preset2 = QtWidgets.QAction(MainWindow)
        self.preset2.setObjectName("preset2")
        self.manualEnable = QtWidgets.QAction(MainWindow)
        self.manualEnable.setCheckable(True)
        self.manualEnable.setObjectName("manualEnable")
        self.menu.addAction(self.loadImages)
        self.menu.addAction(self.clearImages)
        self.menu.addAction(self.saveImg)
        self.menu.addSeparator()
        self.menu.addAction(self.exit)
        self.editMode.addAction(self.editEnable)
        self.debugMode.addAction(self.debugEnable)
        self.analyticalMode.addAction(self.analyticEnable)
        self.presets.addAction(self.preset1)
        self.presets.addAction(self.preset2)
        self.presets.addAction(self.preset3)
        self.presets.addAction(self.preset4)
        self.presets.addAction(self.preset5)
        self.presets.addAction(self.presetDefault)
        self.manualMode.addAction(self.manualEnable)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.editMode.menuAction())
        self.menubar.addAction(self.debugMode.menuAction())
        self.menubar.addAction(self.analyticalMode.menuAction())
        self.menubar.addAction(self.presets.menuAction())
        self.menubar.addAction(self.manualMode.menuAction())

        self.retranslateUi(MainWindow)
        self.comboBox.setCurrentIndex(0)
        self.wave_correct.setCurrentIndex(0)
        self.warp.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "SURF"))
        self.comboBox.setItemText(1, _translate("MainWindow", "ORB"))
        self.comboBox.setItemText(2, _translate("MainWindow", "SIFT"))
        self.comboBox.setItemText(3, _translate("MainWindow", "BRISK"))
        self.comboBox.setItemText(4, _translate("MainWindow", "AKAZE"))
        self.label.setText(_translate("MainWindow", "Алгоритм поиска точек"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p>Порог доверия соответствия точек</p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p>Порог уверенности настроек камеры</p></body></html>"))
        self.wave_correct.setItemText(0, _translate("MainWindow", "Горизонтальная"))
        self.wave_correct.setItemText(1, _translate("MainWindow", "Вертикальная"))
        self.wave_correct.setItemText(2, _translate("MainWindow", "Выключить"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p>коррекция волнового эффекта</p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p>Форма преобразования панорамы</p></body></html>"))
        self.warp.setItemText(0, _translate("MainWindow", "Сферическая"))
        self.warp.setItemText(1, _translate("MainWindow", "Цилиндрическая"))
        self.warp.setItemText(2, _translate("MainWindow", "Аффинная"))
        self.warp.setItemText(3, _translate("MainWindow", "Рыбий глаз"))
        self.warp.setItemText(4, _translate("MainWindow", "Стереографическая"))
        self.warp.setItemText(5, _translate("MainWindow", "paniniA1.5B1"))
        self.warp.setItemText(6, _translate("MainWindow", "paniniA2B1"))
        self.crop.setText(_translate("MainWindow", "Обрезка черных пикселей"))
        self.runButton.setText(_translate("MainWindow", "Создать панораму"))
        self.menu.setTitle(_translate("MainWindow", "Файл"))
        self.editMode.setTitle(_translate("MainWindow", "Режим редактирования"))
        self.debugMode.setTitle(_translate("MainWindow", "Отладочный режим"))
        self.analyticalMode.setTitle(_translate("MainWindow", "Аналитический режим"))
        self.presets.setTitle(_translate("MainWindow", "Предустановленные алгоритмы (пресеты)"))
        self.manualMode.setTitle(_translate("MainWindow", "Ручной режим корректировки"))
        self.dockWidget.setWindowTitle(_translate("MainWindow", "Изображения"))
        self.loadImages.setText(_translate("MainWindow", "Загрузить снимки"))
        self.clearImages.setText(_translate("MainWindow", "Очистить загруженные снимки"))
        self.saveImg.setText(_translate("MainWindow", "Сохранить изображение"))
        self.exit.setText(_translate("MainWindow", "Выход"))
        self.editEnable.setText(_translate("MainWindow", "Включить"))
        self.editDisable.setText(_translate("MainWindow", "Выключить"))
        self.debugEnable.setText(_translate("MainWindow", "Включить"))
        self.analyticEnable.setText(_translate("MainWindow", "Включить"))
        self.preset1.setText(_translate("MainWindow", "Классическая панорама (горизонтальная)"))
        self.preset3.setText(_translate("MainWindow", "Виртуальные туры"))
        self.preset4.setText(_translate("MainWindow", "Виртуальная реальность"))
        self.preset5.setText(_translate("MainWindow", "Архитектурная визуализация"))
        self.presetDefault.setText(_translate("MainWindow", "По умолчанию"))
        self.preset2.setText(_translate("MainWindow", "Классическая панорама (вертикальная)"))
        self.manualEnable.setText(_translate("MainWindow", "Включить"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
