@echo off
title Khoi chay He thong QA - VIQA Nexus
echo ====================================================
echo      DANG KHOI CHAY HE THONG VIQA NEXUS SYSTEM
echo ====================================================
echo.

:: Cau hinh duong dan
set PATH=C:\Program Files\nodejs;%PATH%
set VITE_USE_MOCK_API=false

echo [*] Dang khoi dong Backend API tai cong 8000...
start "VIQA Backend Server" cmd /k "D:\Python\python.exe backend/viqa_api.py"

echo [*] Dang doi Backend khoi dong (5 giay)...
timeout /t 5 /nobreak > nul

echo [*] Dang khoi dong Frontend Web Server tai cong 5173...
start "VIQA Frontend Dev" cmd /k "npm run dev"

echo.
echo ====================================================
echo [OK] Ca hai may chu dang duoc khoi chay!
echo - Web App se mo tai: http://localhost:5173/
echo - Backend API phuc vu tai: http://localhost:8000
echo ====================================================
echo.

:: Tu dong mo trinh duyet sau 3 giay
timeout /t 3 /nobreak > nul
start http://localhost:5173/
