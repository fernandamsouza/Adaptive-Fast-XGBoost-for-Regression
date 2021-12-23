import matplotlib.pyplot as plt

xAxis = [200,400,600,800,1000]
yAxis = [9939.36,9872.88,9899.35,9758.47,9985.02]

yAxisReal = [852.67,847.60,900.53,967.18,1100.65]

yAxis2 = [0,0,0,0,0,0,0,0,0,0,0]

positions = (200, 400, 600, 800 , 1000)
labels = ("200k", "400k", "600k", "800k", "1M")
plt.xticks(positions, labels)
plt.plot(xAxis,yAxis, label='AXGB adaptado', marker='o',)
plt.plot(xAxis,yAxisReal, label='AXGB', marker='v',)
# plt.title('title name')
plt.grid()
plt.legend()

plt.ylim(bottom=0, top=11000)
plt.ylabel('Instâncias por segundo')
plt.xlabel('Tamanho do conjunto (instâncias)')
plt.show()