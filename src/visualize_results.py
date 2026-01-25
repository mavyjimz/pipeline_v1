import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_learning_curve():
    print("[START]: Lesson 23 - Generating AI Learning Report")
    
    # 1. Path Management (Relative to the script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    history_file = r"D:\MLOps\reports\loss_history.csv"
    output_image = r"D:\MLOps\reports\learning_curve.png"

    # 2. Safety Check
    if not os.path.exists(history_file):
        print(f"[ERROR]: History file not found at: {history_file}")
        return

    # 3. Load the data
    df = pd.read_csv(history_file)

    # 4. Create the Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], color='#00ffcc', linewidth=2, label='Training Loss')
    
    # Styling for the Pipeline Warehouse
    plt.title('Superstore Sales AI: Learning Curve (RX 580)', fontsize=14, pad=15)
    plt.xlabel('Epochs (Learning Cycles)', fontsize=12)
    plt.ylabel('Loss (Error Rate)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # Background color adjustment for better visibility
    plt.gca().set_facecolor('#f9f9f9')

    # 5. Save and Show
    plt.savefig(output_image)
    print(f"[SUCCESS]: Report saved to {output_image}")
    plt.show()

if __name__ == "__main__":
    generate_learning_curve()