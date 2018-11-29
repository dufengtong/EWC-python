import matplotlib.pyplot as plt

ewc_accuracy_list = [[0.9583], [0.9366, 0.9586], [0.9367, 0.9488, 0.954]]
no_ewc_accuracy_list = [[0.9558], [0.9116, 0.965], [0.9128, 0.9247, 0.9673]]
    
ax1 = plt.subplot(311)
y1 = [x[0] for x in ewc_accuracy_list ]
y2 = [x[0] for x in no_ewc_accuracy_list ]
ax1.set_ylim([0.8,1])
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(['trainA','trainB','trainC'])
ax1.set_ylabel('TaskA')
ax1.set_xlim([1,3])
ax1.annotate('EWC', xy=(2.85,0.95), color='red')
ax1.annotate('SGD', xy=(2.85,0.88), color='blue')
ax1.plot([1,2,3], y1, 'ro-')
ax1.plot([1,2,3], y2, 'bo-')

ax2 = plt.subplot(312)
y1 = [x[1] for x in ewc_accuracy_list[1:] ]
y2 = [x[1] for x in no_ewc_accuracy_list[1:] ]
ax2.set_ylim([0.8,1])
ax2.set_xlim([1,3])
ax2.set_xticks([1,2,3])
ax2.set_xticklabels(['trainA','trainB','trainC'])
ax2.set_ylabel('TaskB')
ax2.plot([2,3], y1, 'ro-')
ax2.plot([2,3], y2, 'bo-')

ax3 = plt.subplot(313)
y1 = [ewc_accuracy_list[2][-1] ]
y2 = [no_ewc_accuracy_list[2][-1] ]
ax3.set_ylim([0.8,1])
ax3.set_xlim([1,3])
ax3.set_xticks([1,2,3])
ax3.set_xticklabels(['trainA','trainB','trainC'])
ax3.set_ylabel('TaskC')
ax3.plot([3], y1, 'ro-')
ax3.plot([3], y2, 'bo-')

plt.show()