@echo off
set OPENBLAS_NUM_THREADS=4
set KMP_DUPLICATE_LIB_OK=TRUE
echo Starting Khilona Color Detector...
py -3.12 app.py
pause
