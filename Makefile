# btgui Nuitka build
#
# The Windows recipe mirrors the command recorded in PLAN.md; output directory,
# UPX path, icon, console mode, and job count are parameterized below.
#
# Usage:
#   make standalone   # build/nuitka/app.dist/btgui.exe
#   make onefile      # build/nuitka/btgui.exe
#   make clean
#   make help
#
# Requirements:
#   - Python with this project installed editable (pip install -e .)
#   - Nuitka (pip install nuitka) and a working C compiler
#   - GNU Make on Windows (MSYS2, Chocolatey "make", or mingw32-make)
#
# Variables:
#   PYTHON=python JOBS=8 OUTPUT_DIR=build/nuitka CONSOLE_MODE=force
#   ICON=gui_logo.ico UPX=C:/tools/upx.exe
#
# UPX is optional. Windows Defender may quarantine UPX's temporary *.upx files
# and break the build; prefer compressing the finished app.dist with 7z.

PYTHON ?= python
ENTRY ?= src/btgui/app.py
OUTPUT_DIR ?= build/nuitka
JOBS ?= 8
CONSOLE_MODE ?= force
ICON ?= gui_logo.ico
UPX ?=
EXTRA_FLAGS ?=
FULL_VTK ?= 0
PYTHONPATH := src
export PYTHONPATH
# Keep the UPX executable path out of the child environment: UPX itself reads
# a variable named UPX and rejects a Windows path there.
unexport UPX

ifeq ($(OS),Windows_NT)
EXE := .exe
PLATFORM_FLAGS := --windows-console-mode=$(CONSOLE_MODE) --windows-icon-from-ico=$(ICON)
else
EXE := .x
PLATFORM_FLAGS :=
# Conda's libpython*.a is often an LTO/LLVM archive that the system linker
# cannot read ("File format not recognized"). Link the shared libpython instead.
STATIC_LIBPYTHON ?= --static-libpython=no
endif

ifeq ($(strip $(UPX)),)
UPX_FLAGS :=
else
UPX_FLAGS := --plugin-enable=upx --upx-binary=$(UPX)
endif

ifeq ($(FULL_VTK),1)
VTK_FLAGS := --include-package=vtkmodules
else
# Only the modules actually loaded by PyVista and btgui. See vtk_modules.txt;
# --include-package=vtkmodules also pulls the unused VTKm accelerators and
# their ~39 MB VisKores dependency.
VTK_MODULES := $(strip $(file <vtk_modules.txt))
VTK_FLAGS := $(foreach module,$(VTK_MODULES),--include-module=vtkmodules.$(module)) --include-package=vtkmodules.numpy_interface --include-package=vtkmodules.util
endif

COMMON_FLAGS = \
	--enable-plugin=pyside6 \
	--include-qt-plugins=sensible \
	--include-package=btgui \
	--include-package=scipy.optimize \
	--include-package=pyvista.plotting \
	--include-package=pyvistaqt \
	$(VTK_FLAGS) \
	$(STATIC_LIBPYTHON) \
	--include-data-files=src/btgui/settings.json=btgui/settings.json \
	--include-data-dir=src/btgui/data=btgui/data \
	--include-data-dir=src/btgui/frontend/locales=btgui/frontend/locales \
	--output-dir=$(OUTPUT_DIR) \
	--output-filename=btgui$(EXE) \
	--jobs=$(JOBS) \
	--lto=no \
	$(PLATFORM_FLAGS) \
	$(UPX_FLAGS) \
	$(EXTRA_FLAGS)

.PHONY: all standalone onefile clean help

all: standalone

standalone:
	$(PYTHON) -m nuitka --mode=standalone $(COMMON_FLAGS) $(ENTRY)

onefile:
	$(PYTHON) -m nuitka --mode=onefile $(COMMON_FLAGS) $(ENTRY)

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('$(OUTPUT_DIR)', ignore_errors=True)"

help:
	@echo "btgui Nuitka build"
	@echo "  make standalone   standalone directory: $(OUTPUT_DIR)/app.dist/btgui$(EXE)"
	@echo "  make onefile      single executable: $(OUTPUT_DIR)/btgui$(EXE)"
	@echo "  make clean        remove $(OUTPUT_DIR)"
	@echo "  make help         show this text"
	@echo ""
	@echo "Variables: PYTHON=... JOBS=8 OUTPUT_DIR=build/nuitka"
	@echo "           CONSOLE_MODE=force ICON=gui_logo.ico UPX=C:/tools/upx.exe"
	@echo "           FULL_VTK=0 (set 1 for the complete VTK package)"
	@echo "           EXTRA_FLAGS=\"--include-package=scipy\""
