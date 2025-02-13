Colab

simple.ipynb 
This was the exampe code ran in colab. This was used as a benchmark to comfirm the correct versions of libraries and dependciens were  

SimpleRFwithmoDataSave.ipynb
This file built on the SQL injection attack example but replaced the example LSTM model with the tensorflow decison tree (TFDF) Random Forest model. This did not save the RF model to the flatbuffer errors but was used to anaylse the models performace.  

Simpleldata.ipynb
This notebook contains a number of exploritory pieces of code, these are decribed below
Notebook 1
This notebook provides analyse on the input dataset, that is used to test and train the LSTM binary Classifier. the output is to check the number of attack and non attack classifications. It loads the tflite model created by notebook X. It then passes the daatset on sample at the time through the tflite imported model. If the output probablity of the dataset sample being attack is 0.5 is classifies it as an attack (1) and if it less than 0.5 it classifies it as non-attack (0). The number of output predicted attacks and non-attacks produced by the tflite model are printed. This allows for the analyses of the iput and output, and it also for the 0.5 atack classification to be tuned to a percentacge to give the best performamce 
Notebook 2 This notebook is copy of the example trained with more data and it created a tflite mode named lclassifier1.model, this model is used in snort.
Notebook 3 This code snipet creates a LSTM binary Classifcation model and exports this to a tflite model. It used the Westermo input dataset so can be used for comparison of RF and LSTM for bianry classifiaction. It has the advatage that it can be exported to tflite model format, so could by used by snort once an inspector for this type of attack is supported.
Notebook 4 This notebook reads in Tflite model lclassifer3 and provides anaylse on the number of attackes on the input data and then prodcues an output array of the probablitlyes that each daatset sample is an attack.  
Notebook 5 This notebook loads a model and counst the number of attacks the model prodcudes for an imput dataset
