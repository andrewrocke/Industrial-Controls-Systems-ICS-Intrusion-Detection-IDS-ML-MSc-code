**IDS ICS ML**

**Introduction**

This repository contains code and results produced while investigating machine learning for Intrusion Detection Systems (IDS) for Industrial Control Systems (ICS)
The code experiments with 3 implementations of Random Forest Model using

[YDF](https://github.com/google/yggdrasil-decision-forests/) </br>
[Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  </br>
[TensorFlow](https://www.tensorflow.org/decision_forests)

It uses an Industrial network traffic data set from [Westermo](https://github.com/westermo/network-traffic-dataset/)

The work evaluates [Snort3](https://github.com/snort3/snort3) both from a signature-based and machine learning perspective 


It basis the Snort machine learning evaluation on the well described [snortmldocker](https://github.com/ettorecalvi/snortml2docker)

**Setup**

The main Software, libraries version used in the project are shown in the diagram

![Setup](https://github.com/user-attachments/assets/a6aabedc-20d0-4747-a256-bcf25ee4b235)


**Folder Structure**

The folders are decribed as below structure 

- Diary - Contains Supervisor meeting diarays in RTF and in PDF format.
- LSTM - Contains code and results of the SQL Server esample and using the LSTM models with Westermo data, it contains some of the TFLite exported models.
- ML Test - is a scratch pad working folder
- ML - Is the main folder with the machine leraning implmentations
  -   Colab - Contains a readme with some information about the notebooks. The TFDF Random Forest contain results in the reflevance folders and the code in the code folder. 
  -  SkLearn - Contains Default code and code wusing the Hyperparamter Tuning Functions, each folder contains the results organinsed in the dataset folders.
  - YDF - contains the rulseset contained in the dataset folder and it also contains the GradeintBooster Notebook containing a Gradeint booster noteput with code and results.    

The files have the broad naiming convention

Files named Output are the results of code either ran in Python or Notepaks ran in Google Colab

Folders names Left, Right, Bottom contain reaults of using this datset file.

