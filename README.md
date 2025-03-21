**IDS ICS ML**

**Introduction**

This repository contains code and results produced while investigating machine learning for Intrusion Detection Systems (IDS) for Industrial Control Systems (ICS)
The code experiments with 3 implementations of Random Forest Model using

[YDF](https://github.com/google/yggdrasil-decision-forests/) </br>
[Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  </br>
[TensorFlow](https://www.tensorflow.org/decision_forests)

It uses an Industrial network traffic data set from [Westermo](https://github.com/westermo/network-traffic-dataset/)

The work evaluates [Snort3](https://github.com/snort3/snort3) both from a signature-based and machine learning perspective 


It bases the machine learning evaluation on the well described [snortmldocker](https://github.com/ettorecalvi/snortml2docker)
