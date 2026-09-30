
============================================================
EDGE-BASED MACHINE HEALTH MONITORING USING MULTI-SENSORS
============================================================

PROJECT OVERVIEW
----------------
This project investigates machine health monitoring using
signal-processing and machine-learning techniques.

The project analyzes three datasets:

1. MIMII
2. CWRU Bearing Dataset
3. Intelligent Bearing (IB) Dataset

The main machine-learning experiment was performed using
the public IB dataset.


DATASETS
--------
MIMII
- Data type: Machine audio
- Sampling rate used: 16 kHz
- Samples used: 4
- Purpose: EDA and reference

CWRU
- Data type: Bearing vibration
- Sampling rate used: 12 kHz
- Samples used: 4
- Purpose: EDA and reference

IB
- Data type: Bearing vibration
- Sampling rate: 17.85 kHz
- Samples: 178
- Classes: 9
- Purpose: Machine-learning model development


FEATURE EXTRACTION
------------------
Time-domain features:
- Mean
- Standard Deviation
- Minimum
- Maximum
- RMS

Frequency-domain features:
- Dominant Frequency
- Maximum FFT Magnitude
- Spectral Energy
- Spectral Centroid

Total final features: 9


MACHINE LEARNING MODEL
----------------------
Algorithm:
Random Forest Classifier

Number of trees:
100

Random state:
42

Final feature set:
5 time-domain features + 4 FFT features


MODEL RESULTS
-------------
Training Accuracy:
100.00%

Validation Accuracy:
94.44%

Held-out Test Accuracy:
94.44%

5-Fold Cross-Validation Mean:
97.76%

5-Fold Cross-Validation Standard Deviation:
2.08%

Minimum CV Accuracy:
94.44%

Maximum CV Accuracy:
100.00%


FFT FEATURE CONTRIBUTION
------------------------
5-feature model CV accuracy:
94.37%

9-feature model CV accuracy:
97.76%

Improvement:
Approximately 3.40 percentage points


IMPORTANT LIMITATIONS
---------------------
1. Only four recordings were used from MIMII.
2. Only four recordings were used from CWRU.
3. The IB dataset contains 178 samples.
4. The current public IB experiment uses a single accelerometer.
5. The complete physical multi-sensor edge system has not yet
   been experimentally validated.
6. Sample-level cross-validation may overestimate generalization
   if samples are correlated by the same physical bearing or run.


FUTURE WORK
-----------
1. Collect data using multiple physical sensors.
2. Integrate vibration, acoustic, temperature and current sensors.
3. Perform sensor-level data fusion.
4. Collect larger datasets under different operating conditions.
5. Use group-aware evaluation based on bearing/run identity.
6. Compare additional machine-learning and deep-learning models.
7. Deploy the model on an edge computing platform.
8. Perform real-time machine health monitoring.
9. Evaluate latency, memory usage and power consumption.
10. Test the system on real industrial equipment.


PROJECT FOLDER STRUCTURE
------------------------
C:\MachineHealthProject

    data\
        MIMII\
        CWRU\
        IB\

    processed\
        MIMII_processed.csv
        CWRU_processed.csv
        IB_processed.csv
        IB_Time_Frequency_Features.csv
        IB_Model_Comparison.csv
        IB_Feature_Importance.csv
        IB_Final_Model_Results.csv
        IB_Classwise_Performance.csv
        IB_5_vs_9_Feature_CV_Comparison.csv
        IB_Final_Feature_Importance.csv
        IB_Final_Model_Selection.csv
        IB_Final_Evaluation_Summary.csv
        Final_Dataset_Results_Summary.csv
        Final_Project_Methodology_Summary.csv
        Report_Ready_Dataset_Results.csv
        Final_Conclusion_and_Limitations.csv

        models\
            IB_RandomForest_Final.joblib
            IB_Final_Features.joblib

    graphs\
        01_IB_Class_Distribution.png
        02_IB_RMS_By_Class.png
        03_IB_Feature_Importance.png
        04_5_vs_9_Feature_CV.png
        05_IB_Final_Confusion_Matrix.png
        06_IB_Final_Model_Accuracy.png


CONCLUSION
----------
The experiment demonstrates that combining time-domain and
frequency-domain signal features can improve machine-condition
classification on the public IB dataset.

The final Random Forest baseline achieved 94.44% accuracy on
the held-out test set and 97.76% mean accuracy across 5-fold
cross-validation.

The current work provides the software and signal-processing
baseline for the proposed edge-based machine health monitoring
system. Future work will focus on multi-sensor data acquisition,
sensor fusion and real-time edge deployment.


============================================================
END OF README
============================================================
