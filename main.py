# -*- coding: utf-8 -*-
# display a qt widget
import sys
from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Qt

from nndesigner import NNDesigner

import logging

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = NNDesigner()
    window.show()
    sys.exit(app.exec())