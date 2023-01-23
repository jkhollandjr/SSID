'''Gather flow pairs whose packets counts are greater than the threshold per window'''
import os
import argparse

def find_key(input_dict, value):
    '''Get keys for a given value'''
    return {k for k, v in input_dict.items() if v == value}

def parse_csv(csv_path, interval, final_names, threshold): #option: 'sonly', 'tonly', 'both'
    '''For each file in directory, get packets in time interval and record if enough packets sent'''

    here_path = csv_path+'inflow'
    there_path = csv_path+'outflow'
    print(here_path, there_path, interval)

    file_names = []
    for txt_file in os.listdir(here_path):
        file_names.append(txt_file)

    for i in range(len(file_names)):
        with open(here_path + '/' + file_names[i]) as f:
            h_lines = []
            full_lines = f.readlines()
            for line in full_lines:
                time = float(line.split('\t')[0])
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                h_lines.append(line)

        with open(there_path + '/' + file_names[i]) as f:
            t_lines = []
            full_lines = f.readlines()
            for line in full_lines:
                time = float(line.split('\t')[0])
                if float(time) > interval[1]:
                    break
                if float(time) < interval[0]:
                    continue
                t_lines.append(line)
        if(len(h_lines) > threshold) and (len(t_lines) > threshold):
            if file_names[i] in final_names.keys():
                final_names[file_names[i]] += 1
            else:
                final_names[file_names[i]] = 1

    for x in final_names:
        print(x, final_names[x])

def create_overlap_window_csv(csv_path, out_path, threshold, interval, num_windows, addnum):
    global final_names
    final_names = {}
    with open(out_path, 'w+') as fw:
        for win in range(num_windows):
            parse_csv(csv_path, [win*addnum, win*addnum+interval], final_names, threshold)
        for name in list(find_key(final_names, num_windows)):
            fw.write(name)
            fw.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=False, default='CrawlE_Proc_Altered/')
    parser.add_argument('--output_path', required=False, default='original_altered.txt')
    parser.add_argument('--threshold', type=int, required=False, default=1)

    args = parser.parse_args()

    create_overlap_window_csv(args.data_path, args.output_path, args.threshold, 5, 11, 2)
