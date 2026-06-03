@echo off
setlocal

if not exist .venv (
  py -3.12 -m venv .venv
)

call .venv\Scripts\activate
pip install -r requirements.txt

if exist .env (
  for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if not "%%a"=="" set %%a=%%b
  )
)

python train.py --recreate-collection
