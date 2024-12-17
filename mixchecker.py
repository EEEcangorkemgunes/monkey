import json



read_file_test = "test.json"
read_file_train = "train.json"

with open(read_file_test, "r") as f:
    test_data = json.load(f)
with open(read_file_train, "r") as f:
    train_data = json.load(f)

for i, test_patch in enumerate(test_data):
    for train_patch in train_data:
        if test_patch["file_name"] == train_patch["file_name"]:
            print(f"Found match for {test_patch['file_name']} at index {i}")