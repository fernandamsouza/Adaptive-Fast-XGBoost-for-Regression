import matplotlib.pyplot as plt

xAxis = [1,5,10,100]

## hyper_f
yAxisADAPTADO = [22.58,7.94,6.41,6.09]
yAxisAXGB = [185.25,26.12,8.09,7.38]

## sea_a
# yAxisADAPTADO = [20.43,6.96,5.18,3.49,3.30]
# yAxisAXGB = [176.34,38.86,20.78,4.54,2.92]

## agr
# yAxisADAPTADO = [26.90,10.17,7.85,6.66,4.92]
# yAxisAXGB = [204.60,52.16,25.29,6.82,4.91]


positions = (1, 5, 10, 100 , 1000)
labels = ("1", "5", "10", "100", "1000")
plt.xticks(positions, labels)
plt.plot(xAxis,yAxisADAPTADO, label='AXGB adaptado', marker='o',)
plt.plot(xAxis,yAxisAXGB, label='AXGB', marker='v',)
plt.title('hyper_f')
plt.grid()
plt.legend()

# plt.ylim(bottom=0, top=200)
plt.ylabel('Tempo total (segundos)')
plt.xlabel('Batch Size')
plt.show()