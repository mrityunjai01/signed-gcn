import random
import sys
import os
dataset_csv_file = sys.argv[1]
out_file_name = sys.argv[2]
number_of_nodes = sys.argv[3]
split = [0.9, 0.05, 0.05]

#A = numpy.zeros((n,n), dtype=int)

fp = open(dataset_csv_file)
train_file = open(f'{out_file_name}_train.txt', 'w')
val_file = open(f'{out_file_name}_val.txt', 'w')
test_file = open(f'{out_file_name}_test.txt', 'w')
train_data = []
val_data = []
test_data = []
for l in fp:
    R = random.random()
    l = l.split()
    sign = float(l[2])
    if (sign > 0):
        l[2] = '1'
    else:
        l[2] = '-1'
    l = (' ').join(l)
    if (R < split[0]):
        train_data.append(l)

    elif (R < split[0]+split[1]):
        val_data.append(l)
    else:
        test_data.append(l)

train_file.write(f"{number_of_nodes} {len(train_data)}\n")
for line in train_data:
    train_file.write(f'{line}\n')
train_file.close()

val_file.write(f"{number_of_nodes} {len(val_data)}\n")
for line in val_data:
    val_file.write(f'{line}\n')
val_file.close()

test_file.write(f"{number_of_nodes} {len(test_data)}\n")
for line in test_data:
    test_file.write(f'{line}\n')
test_file.close()

fp.close()
