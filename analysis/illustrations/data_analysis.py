import matplotlib.pyplot as plt

mult_40_tpr = []
mult_40_fpr = []
for line in open("capture_results/capture_mult_40.csv", "r"):
    _, tpr, fpr, _ = line.split(',')
    mult_40_tpr.append(float(tpr))
    mult_40_fpr.append(float(fpr))

mult_20_tpr = []
mult_20_fpr = []
for line in open("capture_results/capture_mult_20.csv", "r"):
    _, tpr, fpr, _ = line.split(',')
    mult_20_tpr.append(float(tpr))
    mult_20_fpr.append(float(fpr))

mult_20_fpr = mult_20_fpr[:-4]
mult_20_tpr = mult_20_tpr[:-4]

mult_10_tpr = []
mult_10_fpr = []
for line in open("capture_results/capture_mult_10.csv", "r"):
    _, tpr, fpr, _ = line.split(',')
    mult_10_tpr.append(float(tpr))
    mult_10_fpr.append(float(fpr))


mult_1_tpr = []
mult_1_fpr = []
for line in open("capture_results/capture_mult_1.csv", "r"):
    _, tpr, fpr, _ = line.split(',')
    mult_1_tpr.append(float(tpr))
    mult_1_fpr.append(float(fpr))

mult_1_tpr = mult_1_tpr[:-6]
mult_1_fpr = mult_1_fpr[:-6]

plt.title("DCF attack effectiveness by window adjustment")

nn_fpr = [0.003871871871871872, 0.002242242242242242, 0.0011461461461461462, 0.0005425425425425425, 0.0002192192192192192, 7.107107107107108e-05, 1.3013013013013012e-05, 1.001001001001001e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
nn_tpr = [0.987, 0.972, 0.954, 0.904, 0.819, 0.706, 0.517, 0.203, 0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

mult_50_tpr = []
mult_50_fpr = []
for line in open("capture_results/capture_mult_50.csv", "r"):
    _, tpr, fpr, _ = line.split(',')
    mult_50_tpr.append(float(tpr))
    mult_50_fpr.append(float(fpr))

mult_random_tpr = []
mult_random_fpr = []
for line in open("capture_results/capture_mult_random.csv", "r"):
    _, tpr, fpr, _ = line.split(',')
    mult_random_tpr.append(float(tpr))
    mult_random_fpr.append(float(fpr))


plt.plot(nn_fpr, nn_tpr, label='40x adjustment with NN')
plt.plot(mult_40_fpr, mult_40_tpr, label='40x adjustment')
plt.plot(mult_20_fpr, mult_20_tpr, label='20x adjustment')
plt.plot(mult_10_fpr, mult_10_tpr, label='10x adjustment')
plt.plot(mult_1_fpr, mult_1_tpr, label='No adjustment')
plt.plot(mult_50_fpr, mult_50_tpr, label='50x adjustment')
plt.plot(mult_random_fpr, mult_random_tpr, label='Randomly chosen hop')

#plt.plot(double_front_epoch_500_fpr, double_front_epoch_500_tpr, label='Double Front')


plt.xscale('log')

plt.ylabel("True Positive Rate (TPR)")
plt.xlabel("False Positive Rate (FPR)")

plt.legend()

plt.savefig("capture_chart.png")




