from ResNet_classifer import ResNetClassifier

# Instantiate your ResNetClassifier model
model = ResNetClassifier()

# Access the state_dict and collect its keys
state_dict = model.state_dict()
keys_list = list(state_dict.keys())

# Define the file path to save the output
output_file = 'model_keys.txt'

# Save the keys to a text file
with open(output_file, 'w') as f:
    f.write("State Dict Keys:\n")
    for key in keys_list:
        f.write(f"{key}\n")

print(f"State dict keys saved to {output_file}")