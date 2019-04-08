import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

def smooth(data):# data:[x, y]
    x = data[0]
    y = data[1]
    x_new = np.linspace(min(x), max(x), 5000)
    y_new = spline(x, y, x_new)
    x = x_new
    y = y_new


#fcn_label_loss = [np.arange(0, 105, 5), [0, 0.7839564,0.8123268,0.8193009,0.83644015,0.8357696,0.8392459,0.8323852,0.8406025,0.8399204,0.83450216,0.835337,0.83760154,0.8360845,0.835235,0.83302516,0.83740705,0.82963526,0.8370507,0.8362107,0.8352947]]

fcn_label_loss = [np.arange(0, 35, 5), [0, 0.7839564,0.8123268,0.8193009,0.83644015,0.8357696,0.8392459]]
smooth(fcn_label_loss)

fcn = [np.arange(0, 55, 5), [0, 0.79855704,0.8115551,0.8245252,0.8236592,0.8330308,0.8341005,0.83782226,0.8385549,0.83558565,0.83947127]]
smooth(fcn)

deeplab_resnet = [np.concatenate([np.arange(0, 180, 10), np.arange(180, 210, 5)]), [0, 0.48423776,0.4923622,0.598306,0.65658265,0.6909802,0.72391915,0.74699867,0.76609325,0.7808347,0.79252946,0.8024874,0.81034255,0.8174225,0.8238385,0.82896996,0.8334507,0.8373197,0.8406777,0.9001074,0.90362823,0.9065951,0.9044306,0.90448344]]
smooth(deeplab_resnet)

deeplab_resnet_label_loss = [np.arange(0, 110, 5), [0, 0.44770357,0.4905743,0.5238696,0.81203496,0.84298027,0.8630444,0.8850012,0.88279223,0.889433,0.8940345,0.9027071,0.90439665,0.9075668,0.911924,0.9094226,0.90340126,0.9074497,0.91259396,0.9144217,0.91312796,0.9148634]]
smooth(deeplab_resnet_label_loss)


#deeplab_label_loss = [np.arange(0, 160, 5), [0, 0.4535716,0.48691517,0.50855005,0.51579255,0.53931236,0.794132,0.79252875,0.86712337,0.8695061,0.8719754,0.88343453,0.8968098,0.89305794,0.88755214,0.89736044,0.9028795,0.8900652,0.8962283,0.8983562,0.8943954,0.8927886,0.890374,0.8921598,0.8946565,0.8970221,0.8967752,0.88714206,0.89035755,0.88973165,0.8868756,0.885309]]
deeplab = [np.arange(0, 135, 5), [0, 0.4556294,0.4870176,0.50166196,0.5084013,0.5028152,0.47809768,0.533369,0.768653,0.81179196,0.7893785,0.83372664,0.8438432,0.84023523,0.83349097,0.842417,0.8487067,0.86079824,0.8706472,0.87745476,0.8799504,0.87105376,0.8734948,0.8883561,0.8784894,0.8788755,0.892897
]]
smooth(deeplab)

deeplab_label_loss = [np.arange(0, 85, 5), [0, 0.4535716,0.48691517,0.50855005,0.51579255,0.53931236,0.794132,0.79252875,0.86712337,0.8695061,0.8719754,0.88343453,0.8968098,0.89305794,0.88755214,0.89736044,0.9028795]]
smooth(deeplab_label_loss)


x_ticks = np.arange(0, 220, 20)
plt.xticks(x_ticks)
y_ticks = np.arange(0, 1, 0.1)
plt.yticks(y_ticks)

plt.xlabel('iterations(k)')
plt.ylabel('mIoU in training mode on VOC Val')

plt.plot(fcn[0], fcn[1], label='fcn', linewidth=2)
plt.plot(fcn_label_loss[0], fcn_label_loss[1], label='fcn_label_loss', linewidth=2)
plt.plot(deeplab_resnet[0], deeplab_resnet[1], label='deeplab_resnet', linewidth=2)
plt.plot(deeplab_resnet_label_loss[0], deeplab_resnet_label_loss[1], label='deeplab_resnet_label_loss', linewidth=2)
plt.plot(deeplab[0], deeplab[1], label='deeplab', linewidth=2)
plt.plot(deeplab_label_loss[0], deeplab_label_loss[1], label='deeplab_label_loss', linewidth=2)

plt.legend(['fcn', 'fcn_label_loss', 'deeplab_resnet', 'deeplab_resnet_label_loss', 'deeplab', 'deeplab_label_loss', ])
plt.show()