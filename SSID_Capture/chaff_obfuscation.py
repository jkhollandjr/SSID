import os
import numpy as np
import pandas as pd

def add_dummy_traffic(filepath, outputpath):
    # Read the original file
    df = pd.read_csv(filepath, sep='\t', header=None, names=['time', 'size'])
    
    # Separate packet sizes by direction
    upload_sizes = df[df['size'] > 0]['size'].values
    download_sizes = df[df['size'] < 0]['size'].values
    
    # Generate dummy traffic
    dummy_delays = []
    dummy_sizes = []
    total_time = 0
    while total_time < 20:  # Add dummy packets for the first 20 seconds
        delay = np.random.uniform(0, 0.2)
        direction = np.random.choice([-1, 1])
        if direction > 0 and len(upload_sizes) > 0:
            size = np.random.choice(upload_sizes)
        elif direction < 0 and len(download_sizes) > 0:
            size = np.random.choice(download_sizes)
        else:
            continue
        dummy_delays.append(delay)
        dummy_sizes.append(size)
        total_time += delay
    dummy_times = np.cumsum(dummy_delays)  # Calculate the times of dummy packets
    df_dummy = pd.DataFrame({'time': dummy_times, 'size': dummy_sizes})
    
    # Merge the original and dummy data, and sort by time
    df_total = pd.concat([df, df_dummy])
    df_total.sort_values('time', inplace=True)
    
    # Write the obfuscated data to the output file
    df_total.to_csv(outputpath, sep='\t', header=False, index=False)

def obfuscate_directory(input_dirpath, output_dirpath):
    for filename in os.listdir(input_dirpath):
        input_filepath = os.path.join(input_dirpath, filename)
        output_filepath = os.path.join(output_dirpath, filename.replace('.txt', '_obfuscated.txt'))
        add_dummy_traffic(input_filepath, output_filepath)

# The directories
input_dirpath = '/home/james/Desktop/research/SSID/SSID_Capture/inflow/'
output_dirpath = '/home/james/Desktop/research/SSID/SSID_Capture/inflow_obfuscated_double/'  # Replace with your output directory

obfuscate_directory(input_dirpath, output_dirpath)

