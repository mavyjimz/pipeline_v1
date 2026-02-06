import sys
import time
import winsound

def type_message(message, delay=0.06):
    """Prints message character-by-character with a keyboard click sound."""
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        # Frequency 1200Hz, Duration 10ms for a sharp 'click'
        winsound.Beep(1200, 10) 
        time.sleep(delay)
    print()