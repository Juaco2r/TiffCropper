@@echo off
echo ===============================
echo Building TiffCropper EXE
echo ===============================

REM Activate venv (optional)
call venv\Scripts\activate

REM Clean previous builds
rmdir /s /q build
rmdir /s /q dist

REM Build using spec file
pyinstaller TiffCropper.spec

echo ===============================
echo BUILD COMPLETE
echo Output in /dist
echo ===============================

pause