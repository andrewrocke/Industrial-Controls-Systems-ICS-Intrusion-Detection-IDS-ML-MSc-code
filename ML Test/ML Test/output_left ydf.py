import ydf
import pandas as pd

dataset = pd.read_csv("ML Test\output_left.csv")

model = ydf.GradientBoostedTreesLearner(label="IT_B_Label").train(dataset)

#print(model.describe())
print(model.evaluate(model))

model.save("my_model")

loaded_model = ydf.load_model("my_model")