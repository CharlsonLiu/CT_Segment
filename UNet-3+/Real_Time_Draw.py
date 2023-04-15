import matplotlib.pyplot as plt
import csv
import numpy as np

Unet3pLog_path = 'train_result_2.csv'


with open(Unet3pLog_path) as f:
    f_csv = csv.reader(f)
    head = next(f_csv)  # skip the header row
    epochs = len(list(f_csv))

    # Create a new figure and axis
    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), dpi=80)

    # Set the x and y limits for the plot
    ax.set_xlim([0, epochs])
    ax.set_ylim([0, 1])


# Create four lines for the plot
line1, = ax.plot([], [], 'b.-', linewidth=1.5, label='IoU')
line2, = ax.plot([], [], 'g.-', linewidth=1.5, label='Val IoU')
line3, = ax.plot([], [], 'r.-', linewidth=1.5, label='Loss')
line4, = ax.plot([], [], 'k.-', linewidth=1.5, label='Val Loss')
ax.legend(loc='upper right')
plt.title('FCN IoU and Loss')

def draw(path):
    epochs = []
    loss = []
    iou = []
    val_loss = []
    val_iou = []

    with open(path) as f:
        f_csv = csv.reader(f)

        head = next(f_csv)
        for row in f_csv:
            epochs.append(int(row[0]))
            loss.append(float(row[1]))
            iou.append(float(row[2]))
            val_loss.append(float(row[3]))
            val_iou.append(float(row[4]))

            # Update the data for the four lines
            line1.set_xdata(np.arange(len(iou)))
            line1.set_ydata(np.array(iou))
            line2.set_xdata(np.arange(len(val_iou)))
            line2.set_ydata(np.array(val_iou))
            line3.set_xdata(np.arange(len(loss)))
            line3.set_ydata(np.array(loss))
            line4.set_xdata(np.arange(len(val_loss)))
            line4.set_ydata(np.array(val_loss))

            # Redraw the plot
            fig.canvas.draw()

            # Wait for 0.25 seconds before updating the plot
            plt.pause(0.25)

    plt.legend()


if __name__ == '__main__':
    # FCNLog_path
    draw(Unet3pLog_path)
    plt.show(block=True)
