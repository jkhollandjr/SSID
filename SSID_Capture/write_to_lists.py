import os
import random
from load_data import load_data

data = load_data()
def add_tuples(inner_list):
    for i in range(10):
        first_float = 2.5 * (i + 1)
        second_float = -142
        inner_list.append((first_float, second_float))
    inner_list.sort(key=lambda x: x[0])
    return inner_list
#list structure example
list_of_lists = [
    [
        [(1.0, 2.0), (3.0, 4.0)],
        [(5.0, 6.0), (7.0, 8.0)],
        [(17.0, 18.0), (19.0, 20.0)]  # This inner list will be ignored
    ],
    [
        [(9.0, 10.0), (11.0, 12.0)],
        [(13.0, 14.0), (15.0, 16.0)],
        [(21.0, 22.0), (23.0, 24.0)]  # This inner list will be ignored
    ]
]

dir1 = 'inflow_nov30'
dir2 = 'outflow_nov30'

# Ensure directories exist
os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)

file_counter = 1

size_ratios = []
time_ratios = []
for idx, outer_list in enumerate(data):
    file_name = str(idx + 1)

    # Choose the first and a random trace from the rest of the outer list
    selected_traces = [outer_list[0], outer_list[-1]]

    for j, inner_list in enumerate(selected_traces):
        inner_list = add_tuples(inner_list)
        # Write to the first directory
        if j == 0:
            with open(os.path.join(dir1, file_name), 'w') as f:
                for tuple_ in inner_list:
                    f.write(f'{tuple_[0]}\t{tuple_[1]}\n')
        # Write to the second directory
        elif j == 1:
            with open(os.path.join(dir2, file_name), 'w') as f:
                for tuple_ in inner_list:
                    f.write(f'{tuple_[0]}\t{tuple_[1]}\n')

