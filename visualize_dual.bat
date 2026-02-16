@echo off
REM Visualize dual-command trained model

call C:\armaan_stuff\Coding\Brain_argo\.venv\Scripts\activate.bat
python visualize.py --exp_name go2-dual-commands --latest --episodes 5 %*
