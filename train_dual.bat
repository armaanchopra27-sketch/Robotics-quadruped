@echo off
REM Train locomotion network with dual commands from checkpoint 900

call C:\armaan_stuff\Coding\Brain_argo\.venv\Scripts\activate.bat
python src\go2_train.py --exp_name go2-dual-commands --checkpoint 900 --load_from go2-walking --steps 3000 --envs 512 %*
