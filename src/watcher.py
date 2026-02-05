import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- THE SENTINEL LOGIC ---
class PipelineTriggerHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # We only care about our raw data file
        if "superstore_sales.csv" in event.src_path:
            print(f"\n[!] ALERT: Change detected in {event.src_path}")
            print("--- Triggering Master Automation Sequence ---")
            
            # This calls your .bat file automatically
            try:
                subprocess.run(["automate_all.bat"], shell=True, check=True)
                print("--- Sequence Complete. Returning to watch mode... ---")
            except Exception as e:
                print(f"Error triggering batch: {e}")

if __name__ == "__main__":
    # Ensure we are watching the correct relative path
    watch_path = "./input_data/raw/"
    
    event_handler = PipelineTriggerHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    
    print("====================================================")
    print(f"SENTINEL ACTIVE: Watching {os.path.abspath(watch_path)}")
    print("Press Ctrl+C to stop the watcher.")
    print("====================================================")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nSentinel standing down.")
    observer.join()