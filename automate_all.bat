@echo off
set LOGFILE=reports\automation_log.txt
echo [%DATE% %TIME%] --- Phase 4: Automation Sequence Initiated --- >> %LOGFILE%

:: 1. Run Phase 1 & 2 Core ML Pipeline
docker exec mlops_pipeline_engine bash ./src/run_pipeline.sh

:: 2. Run Phase 3 Visualizers
docker exec mlops_pipeline_engine python3 src/visualize_results.py
docker exec mlops_pipeline_engine python3 src/dashboard.py

:: NEW: Start the Python Server in a SEPARATE minimized window so it stays open
cd reports
start /min python -m http.server 8000
cd ..

:: 3. Finalization
echo [%DATE% %TIME%] --- Phase 4: Sequence Complete --- >> %LOGFILE%
start chrome "http://localhost:8000"

pause