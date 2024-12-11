import ydf
import pandas as pd

ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"
train_ds = pd.read_csv(f"{ds_path}/adult_train.csv")
test_ds = pd.read_csv(f"{ds_path}/adult_test.csv")

model = ydf.GradientBoostedTreesLearner(label="income").train(train_ds)

print(model.evaluate(test_ds))

model.save("my_model")

loaded_model = ydf.load_model("my_model")