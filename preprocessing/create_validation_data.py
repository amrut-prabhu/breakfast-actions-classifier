import sklearn
import pickle
from sklearn.model_selection import train_test_split

f = open("drive/My Drive/breakfast-actions-classifier/Copy of training_data.p",
"rb")


training_data_inputs = []
training_data_true_outputs = []
counter = 1
while True:
  try:
    (segment, label) = pickle.load(f)
    
    if counter % 100 == 0:
      print(f"at sample: {counter}")
    training_data_inputs.append(segment)
    training_data_true_outputs.append(label)
    counter += 1

  except (EOFError):
    break

f.close()

X_train, X_val, y_train, y_val = train_test_split(training_data_inputs, 
training_data_true_outputs, test_size=0.2, random_state=1)

training_out = open("drive/My Drive/breakfast-actions-classifier/trimmed_training_data.p", 'wb')
validation_out = open("drive/My Drive/breakfast-actions-classifier/validation_data.p", 'wb')

print(len(X_train))
print(len(X_val))

counter = 1
for i, segment  in enumerate(X_train):
  if counter % 100 == 0:
    print(f"dumping sample {counter} in {training_out}") 
  pickle.dump((segment, y_train[i]), training_out)
  counter += 1

counter = 1
for i, segment in enumerate(X_val):
  if counter % 100 == 0:
    print(f"dumping sample {counter} in {validation_out}")
  pickle.dump((segment, y_val[i]), validation_out)
  counter += 1

training_out.close()
validation_out.close()

