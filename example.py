import readData

c = readData.Dataset("part2_training_data.csv")
features, labels = c

print(features.shape)
