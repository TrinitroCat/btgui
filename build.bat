@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem btgui Nuitka build for Windows.
rem Usage:
rem   build.bat standalone
rem   build.bat onefile
rem   build.bat clean
rem   build.bat help
rem
rem Environment overrides:
rem   set PYTHON=python
rem   set JOBS=8
rem   set OUTPUT_DIR=D:\nuitka_build
rem   set CONSOLE_MODE=force
rem   set ICON=gui_logo.ico
rem   set UPX_EXE=C:\path\to\upx.exe
rem   set FULL_VTK=1
rem   set EXTRA_FLAGS=--include-package=scipy
rem
rem UPX is optional: Windows Defender may quarantine UPX's *.upx temp files
rem (for example scipy\special\cython_special.upx) and break the build.

if not defined PYTHON set "PYTHON=python"
if not defined JOBS set "JOBS=8"
if not defined OUTPUT_DIR set "OUTPUT_DIR=build\nuitka"
if not defined CONSOLE_MODE set "CONSOLE_MODE=force"
if not defined ICON set "ICON=gui_logo.ico"
if not defined FULL_VTK set "FULL_VTK=0"
if not defined EXTRA_FLAGS set "EXTRA_FLAGS="
rem Accept the old UPX variable, then remove it: the UPX executable reads its
rem own UPX environment variable and fails on a Windows path.
if not defined UPX_EXE if defined UPX set "UPX_EXE=%UPX%"
set "UPX="
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "ENTRY=src\btgui\app.py"

set "UPX_FLAGS="
if defined UPX_EXE set "UPX_FLAGS=--plugin-enable=upx --upx-binary="%UPX_EXE%""

set "VTK_FLAGS="
if /i "%FULL_VTK%"=="1" (
    set "VTK_FLAGS=--include-package=vtkmodules"
) else (
    for /f "usebackq delims=" %%M in ("vtk_modules.txt") do (
        set "VTK_FLAGS=!VTK_FLAGS! --include-module=vtkmodules.%%M"
    )
    set "VTK_FLAGS=!VTK_FLAGS! --include-package=vtkmodules.numpy_interface --include-package=vtkmodules.util"
)

set "COMMON_FLAGS=--enable-plugin=pyside6 --include-qt-plugins=sensible --include-package=btgui --include-package=scipy.optimize --include-package=pyvista.plotting --include-package=pyvistaqt %VTK_FLAGS% --include-data-files=src\btgui\settings.json=btgui\settings.json --include-data-dir=src\btgui\data=btgui\data --output-dir="%OUTPUT_DIR%" --output-filename=btgui.exe --jobs=%JOBS% --lto=no --windows-console-mode=%CONSOLE_MODE% --windows-icon-from-ico="%ICON%" %UPX_FLAGS% %EXTRA_FLAGS%"

if /i "%~1"=="standalone" goto standalone
if /i "%~1"=="onefile" goto onefile
if /i "%~1"=="clean" goto clean
if /i "%~1"=="flags" goto flags
if /i "%~1"=="help" goto help
if "%~1"=="" goto help
echo Unknown target: %~1
exit /b 1

:standalone
"%PYTHON%" -m nuitka --mode=standalone %COMMON_FLAGS% "%ENTRY%"
if errorlevel 1 goto fail
echo Built %OUTPUT_DIR%\app.dist\
goto end

:onefile
"%PYTHON%" -m nuitka --mode=onefile %COMMON_FLAGS% "%ENTRY%"
if errorlevel 1 goto fail
echo Built %OUTPUT_DIR%\btgui.exe
goto end

:clean
if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"
echo Cleaned %OUTPUT_DIR%
goto end

:flags
echo %COMMON_FLAGS%
goto end

:help
echo btgui Nuitka build
echo.
echo   build.bat standalone   build %OUTPUT_DIR%\app.dist\btgui.exe
echo   build.bat onefile      build %OUTPUT_DIR%\btgui.exe
echo   build.bat clean        remove %OUTPUT_DIR%
echo   build.bat help         show this text
echo.
echo Environment overrides:
echo   set PYTHON=python
echo   set JOBS=8
echo   set OUTPUT_DIR=D:\nuitka_build
echo   set CONSOLE_MODE=force
echo   set ICON=gui_logo.ico
echo   set UPX_EXE=C:\path\to\upx.exe
echo   set FULL_VTK=1
echo   set EXTRA_FLAGS=--include-package=scipy
goto end

:fail
echo Build failed.
exit /b 1

:end
endlocal
exit /b 0
