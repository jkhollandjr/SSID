import os
import random

def add_random_delay(packet_time):
    # Add a random delay between 0 to 1 second (inclusive of 0, exclusive of 1)
    return packet_time + random.uniform(0, 2)

def process_trace_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        print(input_file)
        for line in infile:
            # Split the line at tab character
            parts = line.strip().split('\t')
            
            # Assuming the first column is the packet time
            original_time = float(parts[0])
            delayed_time = add_random_delay(original_time)
            
            # Write the modified time and original direction to the output file
            outfile.write(f"{delayed_time}\t{parts[1]}\n")

def process_directory(input_directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for trace_file in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, trace_file)
        output_file_path = os.path.join(output_directory, trace_file)
        
        if os.path.isfile(input_file_path):
            process_trace_file(input_file_path, output_file_path)

if __name__ == "__main__":
    SOURCE_DIRECTORY = "/home/james/Desktop/research/SSID/SSID_Capture/outflow/"
    TARGET_DIRECTORY = "/home/james/Desktop/research/SSID/SSID_Capture/outflow_delayed_2/"
    
    process_directory(SOURCE_DIRECTORY, TARGET_DIRECTORY)
