import matplotlib.pyplot as plt

xAxis = [1,10,20,30,40,50,60,70,80,90]
yAxis = [84.62,88.20,88.99,89.22,89.35,89.56,89.72,89.93,89.68,89.77]

yAxisReal = [71.00,88.00,88.76,88.95,89.40,89.61,89.75,89.83,89.81,89.76]

yAxis2 = [0,0,0,0,0,0,0,0,0,0,0]

plt.plot(xAxis,yAxis, label='AXGB adaptado',marker='o')
plt.plot(xAxis,yAxisReal, label='AXGB',marker='v')
# plt.title('title name')
plt.grid()
plt.legend()
plt.ylim(bottom=0, top=100)
plt.ylabel('Acurácia')
plt.xlabel('% de dados rotulados')
plt.show()

# import matplotlib.pyplot as plt

# xAxis = [1,10,20,30,40,50,60,70,80,90]
# yAxis = [60.22,69.11,71.93,72.01,73.76,74.09,74.55,73.36,73.56,74.88]

# yAxisReal = [65.12,67.08,67.75,63.62,64.82,71.84,69.68,67.47,67.84,68.97]

# yAxis2 = [0,0,0,0,0,0,0,0,0,0,0]

# plt.plot(xAxis,yAxis, label='AXGB adaptado',marker='o')
# plt.plot(xAxis,yAxisReal, label='AXGB',marker='v')
# # plt.title('title name')
# plt.grid()
# plt.legend()
# plt.ylim(bottom=0, top=100)
# plt.ylabel('Acurácia')
# plt.xlabel('% de dados rotulados')
# plt.show()