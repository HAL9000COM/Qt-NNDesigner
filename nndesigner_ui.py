# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'nndesigner.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QApplication, QComboBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QTextBrowser, QVBoxLayout, QWidget)

class Ui_nndesigner(object):
    def setupUi(self, nndesigner):
        if not nndesigner.objectName():
            nndesigner.setObjectName(u"nndesigner")
        nndesigner.resize(500, 400)
        nndesigner.setMouseTracking(True)
        nndesigner.setAcceptDrops(True)
        nndesigner.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.verticalLayout_2 = QVBoxLayout(nndesigner)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(nndesigner)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.lineEdit_export_path = QLineEdit(nndesigner)
        self.lineEdit_export_path.setObjectName(u"lineEdit_export_path")

        self.horizontalLayout_2.addWidget(self.lineEdit_export_path)

        self.pushButton_browse = QPushButton(nndesigner)
        self.pushButton_browse.setObjectName(u"pushButton_browse")

        self.horizontalLayout_2.addWidget(self.pushButton_browse)

        self.label_3 = QLabel(nndesigner)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_2.addWidget(self.label_3)

        self.lineEdit_export_name = QLineEdit(nndesigner)
        self.lineEdit_export_name.setObjectName(u"lineEdit_export_name")

        self.horizontalLayout_2.addWidget(self.lineEdit_export_name)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(nndesigner)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_3.addWidget(self.label_2)

        self.comboBox_type = QComboBox(nndesigner)
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.setObjectName(u"comboBox_type")

        self.horizontalLayout_3.addWidget(self.comboBox_type)

        self.label_4 = QLabel(nndesigner)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_3.addWidget(self.label_4)

        self.comboBox_dtype = QComboBox(nndesigner)
        self.comboBox_dtype.addItem("")
        self.comboBox_dtype.addItem("")
        self.comboBox_dtype.addItem("")
        self.comboBox_dtype.addItem("")
        self.comboBox_dtype.addItem("")
        self.comboBox_dtype.setObjectName(u"comboBox_dtype")

        self.horizontalLayout_3.addWidget(self.comboBox_dtype)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.tableWidget = QTableWidget(nndesigner)
        if (self.tableWidget.columnCount() < 3):
            self.tableWidget.setColumnCount(3)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        if (self.tableWidget.rowCount() < 2):
            self.tableWidget.setRowCount(2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        __qtablewidgetitem5.setFlags(Qt.ItemIsSelectable|Qt.ItemIsDragEnabled|Qt.ItemIsDropEnabled|Qt.ItemIsUserCheckable|Qt.ItemIsEnabled);
        self.tableWidget.setItem(0, 0, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        __qtablewidgetitem6.setFlags(Qt.ItemIsSelectable|Qt.ItemIsDragEnabled|Qt.ItemIsDropEnabled|Qt.ItemIsUserCheckable|Qt.ItemIsEnabled);
        self.tableWidget.setItem(0, 1, __qtablewidgetitem6)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.tableWidget.setItem(0, 2, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        __qtablewidgetitem8.setFlags(Qt.ItemIsSelectable|Qt.ItemIsDragEnabled|Qt.ItemIsDropEnabled|Qt.ItemIsUserCheckable|Qt.ItemIsEnabled);
        self.tableWidget.setItem(1, 0, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        __qtablewidgetitem9.setFlags(Qt.ItemIsSelectable|Qt.ItemIsDragEnabled|Qt.ItemIsDropEnabled|Qt.ItemIsUserCheckable|Qt.ItemIsEnabled);
        self.tableWidget.setItem(1, 1, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.tableWidget.setItem(1, 2, __qtablewidgetitem10)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setMouseTracking(False)
        self.tableWidget.setAcceptDrops(False)
        self.tableWidget.setAutoFillBackground(True)
        self.tableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
        self.tableWidget.setProperty("showDropIndicator", False)
        self.tableWidget.setDragEnabled(False)
        self.tableWidget.setDragDropOverwriteMode(False)
        self.tableWidget.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.tableWidget.setDefaultDropAction(Qt.IgnoreAction)
        self.tableWidget.setAlternatingRowColors(True)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tableWidget.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setProperty("showSortIndicator", False)

        self.horizontalLayout_4.addWidget(self.tableWidget)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.pushButton_add = QPushButton(nndesigner)
        self.pushButton_add.setObjectName(u"pushButton_add")

        self.verticalLayout.addWidget(self.pushButton_add)

        self.pushButton_delete = QPushButton(nndesigner)
        self.pushButton_delete.setObjectName(u"pushButton_delete")

        self.verticalLayout.addWidget(self.pushButton_delete)

        self.pushButton_up = QPushButton(nndesigner)
        self.pushButton_up.setObjectName(u"pushButton_up")

        self.verticalLayout.addWidget(self.pushButton_up)

        self.pushButton_down = QPushButton(nndesigner)
        self.pushButton_down.setObjectName(u"pushButton_down")

        self.verticalLayout.addWidget(self.pushButton_down)

        self.pushButton_clear = QPushButton(nndesigner)
        self.pushButton_clear.setObjectName(u"pushButton_clear")

        self.verticalLayout.addWidget(self.pushButton_clear)

        self.pushButton_test = QPushButton(nndesigner)
        self.pushButton_test.setObjectName(u"pushButton_test")

        self.verticalLayout.addWidget(self.pushButton_test)

        self.pushButton_export = QPushButton(nndesigner)
        self.pushButton_export.setObjectName(u"pushButton_export")

        self.verticalLayout.addWidget(self.pushButton_export)


        self.horizontalLayout.addLayout(self.verticalLayout)


        self.horizontalLayout_4.addLayout(self.horizontalLayout)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.textBrowser = QTextBrowser(nndesigner)
        self.textBrowser.setObjectName(u"textBrowser")
        font = QFont()
        font.setFamilies([u"Consolas"])
        self.textBrowser.setFont(font)
        self.textBrowser.setAcceptDrops(False)

        self.verticalLayout_2.addWidget(self.textBrowser)


        self.retranslateUi(nndesigner)

        QMetaObject.connectSlotsByName(nndesigner)
    # setupUi

    def retranslateUi(self, nndesigner):
        nndesigner.setWindowTitle(QCoreApplication.translate("nndesigner", u"Neural Network Designer", None))
        self.label.setText(QCoreApplication.translate("nndesigner", u"Export Path:", None))
        self.pushButton_browse.setText(QCoreApplication.translate("nndesigner", u"Open", None))
        self.label_3.setText(QCoreApplication.translate("nndesigner", u"Export name:", None))
        self.label_2.setText(QCoreApplication.translate("nndesigner", u"Export format:", None))
        self.comboBox_type.setItemText(0, QCoreApplication.translate("nndesigner", u"Keras HDF5", None))
        self.comboBox_type.setItemText(1, QCoreApplication.translate("nndesigner", u"Keras SavedModel", None))
        self.comboBox_type.setItemText(2, QCoreApplication.translate("nndesigner", u"ONNX", None))

        self.label_4.setText(QCoreApplication.translate("nndesigner", u"Float:", None))
        self.comboBox_dtype.setItemText(0, QCoreApplication.translate("nndesigner", u"float32", None))
        self.comboBox_dtype.setItemText(1, QCoreApplication.translate("nndesigner", u"float16", None))
        self.comboBox_dtype.setItemText(2, QCoreApplication.translate("nndesigner", u"float64", None))
        self.comboBox_dtype.setItemText(3, QCoreApplication.translate("nndesigner", u"mixed_float16", None))
        self.comboBox_dtype.setItemText(4, QCoreApplication.translate("nndesigner", u"mixed_bfloat16", None))

        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("nndesigner", u"Type", None));
        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("nndesigner", u"Name", None));
        ___qtablewidgetitem2 = self.tableWidget.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("nndesigner", u"Arguments", None));
        ___qtablewidgetitem3 = self.tableWidget.verticalHeaderItem(0)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("nndesigner", u"Input", None));
        ___qtablewidgetitem4 = self.tableWidget.verticalHeaderItem(1)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("nndesigner", u"Output", None));

        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        ___qtablewidgetitem5 = self.tableWidget.item(0, 0)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("nndesigner", u"Input Layer", None));
        ___qtablewidgetitem6 = self.tableWidget.item(0, 1)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("nndesigner", u"None", None));
        ___qtablewidgetitem7 = self.tableWidget.item(0, 2)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("nndesigner", u"2", None));
        ___qtablewidgetitem8 = self.tableWidget.item(1, 0)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("nndesigner", u"Output Layer", None));
        ___qtablewidgetitem9 = self.tableWidget.item(1, 1)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("nndesigner", u"None", None));
        ___qtablewidgetitem10 = self.tableWidget.item(1, 2)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("nndesigner", u"2", None));
        self.tableWidget.setSortingEnabled(__sortingEnabled)

        self.pushButton_add.setText(QCoreApplication.translate("nndesigner", u"Add", None))
        self.pushButton_delete.setText(QCoreApplication.translate("nndesigner", u"Delete", None))
        self.pushButton_up.setText(QCoreApplication.translate("nndesigner", u"\u2191", None))
        self.pushButton_down.setText(QCoreApplication.translate("nndesigner", u"\u2193", None))
        self.pushButton_clear.setText(QCoreApplication.translate("nndesigner", u"Clear", None))
        self.pushButton_test.setText(QCoreApplication.translate("nndesigner", u"Test", None))
        self.pushButton_export.setText(QCoreApplication.translate("nndesigner", u"Export", None))
    # retranslateUi

