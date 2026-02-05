@echo off
set LOGFILE=reports\automation_log.txt
echo [%DATE% %TIME%] --- Phase 4: Automation Sequence Initiated --- >> %LOGFILE%

:: 1. Run Phase 1 & 2 Core ML Pipeline
echo Running Pipeline (Cleaning, Features, Training)...
docker exec mlops_pipeline_engine bash ./src/run_pipeline.sh
if %ERRORLEVEL% EQU 0 (echo [%DATE% %TIME%] SUCCESS: Core Pipeline >> %LOGFILE%) else (echo [%DATE% %TIME%] FAIL: Core Pipeline >> %LOGFILE%)

:: 2. Run Phase 3 Visualizers
echo Generating Visuals and Dashboard JSON...
docker exec mlops_pipeline_engine python3 src/visualize_results.py
docker exec mlops_pipeline_engine python3 src/dashboard.py
echo [%DATE% %TIME%] SUCCESS: Visuals Updated >> %LOGFILE%

:: 3. Finalization
echo [%DATE% %TIME%] --- Phase 4: Sequence Complete --- >> %LOGFILE%

:: NEW: Automatically open the dashboard in Chrome
start chrome "http://localhost:8000"

echo -------------------------------------------------- >> %LOGFILE%
echo AUTOMATION COMPLETE: Dashboard launched in Chrome.
pause