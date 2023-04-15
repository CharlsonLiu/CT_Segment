import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.animation as animation

Weighted_path = 'unet-(pp)/unet2p_iou_values_2.csv'

def update(epoch):
    epochs = []
    loss = []

    with open(Weighted_path) as f:
        f_csv = csv.reader(f)

        head = next(f_csv)
        for row in f_csv:
            epochs.append(int(row[0]))
            loss.append(float(row[1]))

    # Clear the current axis
    plt.cla()

    # Create four lines for the plot

    line3, = ax.plot(epochs[:epoch], loss[:epoch], 'b.-', linewidth=1.5, label='Weighted_IoU')

    # Set the x and y limits for the plot
    ax.set_xlim([0, max(epochs)])
    ax.set_ylim([0, 1])

    # Add title and legend
    ax.set_title('Weighted Test IoU')
    ax.legend(loc='upper right')

    mean_iou = np.mean(loss)
    var_iou = np.var(loss)
    # 添加均值和方差到文本框
    textstr = '\n'.join((
        r'$\mathrm{Mean\ IoU}=%.5f$' % (mean_iou,),
        r'$\mathrm{Var\ IoU}=%.5f$' % (var_iou,)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)

    return line3

if __name__ == '__main__':
    # Create a new figure and axis
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=80)

    # Create a FuncAnimation object and save as gif
    anim = animation.FuncAnimation(fig, update, frames=len(list(csv.reader(open(Weighted_path)))), repeat=False)
    anim.save('Weighted Test IoU.gif', writer='pillow')

    # Show the plot
    plt.show()
