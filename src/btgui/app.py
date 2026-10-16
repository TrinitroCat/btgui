"""Application entry point."""

import sys

from PySide6.QtCore import QCommandLineParser, QTimer
from PySide6.QtWidgets import QApplication

from btgui.frontend.main_window import MainWindow
from btgui.integrations.structure_io import make_structure_handlers


def main():
    """Run the Qt application and open optional positional structure files.

    Args:
        None.

    Return:
        The QApplication exit code.

    Note:
        Accepts ``btgui [files...]``; Qt retains its own application options.
        Shell quoting preserves spaces within each file path.
    """
    application = QApplication(sys.argv)
    # Section: Parse positional paths after Qt consumes its own options. A
    # literal '--' allows paths beginning with '-' to remain positional files.
    parser = QCommandLineParser()
    parser.setApplicationDescription("View molecular and crystal structure files.")
    parser.addHelpOption()
    parser.addPositionalArgument("files", "Structure files to open at startup.", "[files...]")
    parser.process(application)
    files = parser.positionalArguments()
    open_handler, save_handler = make_structure_handlers()
    window = MainWindow(open_handler=open_handler, save_handler=save_handler)
    window.show()
    if files:
        # Defer rendering-dependent loading until the Qt event loop starts.
        # Reuse normal file loading and its GUI-console error reporting.
        QTimer.singleShot(0, lambda: window._open_dropped_files(files))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
