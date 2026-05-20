@echo off
echo ===============================
echo Building TiffCropper EXE
echo ===============================

REM Activate venv
if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo WARNING: venv not found. Using current Python environment.
)

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build using spec file
pyinstaller --clean --noconfirm TiffCropper.spec

echo ===============================
echo BUILD COMPLETE
echo Output in /dist
echo ===============================

pause