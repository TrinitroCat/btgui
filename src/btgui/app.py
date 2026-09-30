"""Application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from btgui.frontend.main_window import MainWindow
from btgui.integrations.structure_io import make_structure_handlers


def main():
    """Create and run the Qt application.

    Args:
        None.

    Return:
        The QApplication exit code.
    """
    application = QApplication(sys.argv)
    open_handler, save_handler = make_structure_handlers()
    window = MainWindow(open_handler=open_handler, save_handler=save_handler)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
