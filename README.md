# Predictive-Anomaly-Detection-in-NGAFID-Data-Using-Neuroevolution-and-Deep-Learning
Step 1: Create a venv
    Create a virtual environment. 
    On macOS or Linux, use the command:
    python3 -m venv myenv
    source myenv/bin/activate

    On Windows, use the command:
    python -m venv myenv
    myenv\Scripts\activate


Step 2: Download Requirements
    Download the required dependencies by running 
    pip install -r requirements.txt

    This will install all necessary Python packages such as numpy, pandas, scikit-learn, and matplotlib.

    Make sure to place your raw .csv flight files inside the dataset/ directory. 
    The script automatically parses filenames to distinguish between before and after maintenance cases. This project is cross-platform and runs on both Mac and Windows. If you're unsure where to start, begin with ocsvm_pipeline2.py to test anomaly detection on a small subset of your dataset.

Step 3: Train the Model
    Train the model by running ocsvm_pipeline2.py file. 
    python ocsvm_pipeline2.py

    This script will load flight logs from the dataset/ folder, clean the data, extract overlapping windows of sensor readings, and train an OC-SVM model using only post-maintenance flights as the normal class. Pre-maintenance flights are used for evaluation. The model will output precision, recall, and F1-score for anomaly detection. 

Step 4: Detect Anomalies
    Detect anomalies by running detect_anomalies.py file. 
    python detect_anomalies.py

    This script allows you to apply the trained model to additional flight data, flagging suspicious windows that may indicate faults. You can customize the window size, overlap, or scoring thresholds directly in the script.
    Notes for Customization:
        Adjust window size: Edit WINDOW_SIZE = 100 and STEP_SIZE = 50 in ocsvm_pipeline2.py
        Use more files: Remove after_files = after_files[:100] for full training
        Include day-of: If needed, edit the file loader in load_csv_files()

Step 5: Run the EXAMM Model for accurate anomaly detection