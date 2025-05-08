import os
import pickle
import pandas as pd

keypoints = [5, 9, 13]
directories = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('depth_dataset')]
all_data = []

for directory in directories:
    distance = directory.split('_')[-1]
    print(f"Directory: {directory}, Distance: {distance}")

    pkl_files = [f for f in os.listdir(directory) if f.startswith('image_hand') and f.endswith('.pkl')]

    for pkl_file in pkl_files:
        data=pickle.load(open(os.path.join(directory, pkl_file), 'rb'))
        print(pkl_file)
        data = data - data[0]
        data = data[keypoints].flatten()
        print(data,distance)
        all_data.append([data.tolist(), distance])
df = pd.DataFrame(all_data, columns=['Data', 'Distance'])
df.to_csv('output_data.csv', index=False)
print("Data has been saved to 'output_data.csv'.")
