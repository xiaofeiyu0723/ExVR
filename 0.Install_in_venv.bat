

echo Installing virtualenv...
python -m pip install virtualenv
echo Installation complete
virtualenv venv
call venv\Scripts\activate
pause

echo Entered venv environment
pause
echo Installing the dependency packages required for ExVR...
pip install -r requirements.txt
echo Installation complete
pause 

echo If you didn't installed Microsoft Visual C++.
echo You shoud view this website to download and install it.
echo https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

pause 
