import numpy as np
import matplotlib.pyplot as plt

peak_input = 8
threshold_input = 0.1
data = np.loadtxt("Data FrequenzH10.4.txt")
f = data[:, 0]
a = data[:, 1]

def sinusoid(A, F, fs=16000, t=1.0):
  ts = np.arange(0.0, t, 1.0 / fs)
  return A * np.sin(2 * np.pi * ts * F)

def linear(m, x1, y1):
  x = (m*x1 - y1)/m
  return x

for i in range(len(a)):
    a[i] = 10**(a[i]/20)

x = 0
for i in range(len(a)):
    if a[x] <= a[i]:
        a_Max = a[i]
        x = i

for i in range(len(a)):
    a[i] = a[i]/a_Max

threshold = threshold_input
for i in range(len(a)):
    if a[i] < threshold:
       a[i] = 0.0

peak = peak_input              #Abstand(*21.5) zwischen zwei Peaks im Frequenzspektrum
for i in range(len(a)-1):
    for j in range(1, peak):
        if a[i-j] < a[i] > a[i+j]:
            a[i] = a[i]
        else:
            a[i] = 0

#base = False
#for i in range(len(a)):
    #if a[i] != 0 and base == False:
        #a[i] = a[i]
        #base = True
    #else:
        #a[i] = 0
T = 1                   #Periodendauer von den Peaks und somit der Grundfrequenz
for i in range(len(a)):
    if a[i] != 0:
        T = 1/f[i]
        break
print("Periodendauer T:", T)

N = 0
for i in range(len(a)):
    if a[i] != 0:
        print(f[i], a[i])
        N = N + 1
print("Anzahl Obertöne:", N-1)

#signal = sinusoid(0.0, 0.0)
#for i in range(len(a)):
    #if a[i] != 0:
        #signal = signal + sinusoid(a[i], f[i])

#plt.figure(figsize=(16, 6))
#plt.plot(signal[:1000])

plt.figure()
plt.xlim(0,10000)
plt.plot(f, a)
plt.show()

y = np.fft.ifft(a)
for i in range(len(y)):
    y[i] = y[i].real
#y = np.abs(y)

b = 20      #Beginning and End of specified Area where Peak is
e = 100
w = b
for i in range(b, e):
    if y[w] <= y[i]:
        x_Maximum = i
        y_Maximum = y[i]
        w = i
print("x_Max and y_Max:", x_Maximum, y_Maximum)

y_sum = 0
for i in range(len(y)):
    y_sum = y_sum + y[i]
y_average = y_sum/len(y) + 0.001
print("Average of y-axis:", y_average)

deltaT_threshold = (y_Maximum + y_average)/2           #y-Coordinate"y1 and y2" where we can measure the distance of delta T
print("deltaT_threshold:", deltaT_threshold)

y_Max = y_Maximum
x_Max = x_Maximum
for i in range(b, x_Maximum):
    if y[i] >= deltaT_threshold:
        y_Max = y[i]
        x_Max = i

print("Point over threshold on the left:", x_Max, y_Max)

for i in range(b, x_Max):
    if y[i] <= deltaT_threshold:
        y_Min = y[i]
        x_Min = i

print("Point under threshold on the left:", x_Min, y_Min)

ratio = (deltaT_threshold - y_Min)/(y_Max - y_Min)
x1 = ratio*(x_Max - x_Min) + x_Min
print("X-Coordinate left side:", x1)

key = False
for i in range(x_Max, e):
    if y[i] <= deltaT_threshold and key == False:
        y_Min2 = y[i]
        x_Min2 = i
        key = True
print("Point under threshold on the right:", x_Min2, y_Min2)

for i in range(x_Max, x_Min2):
    if y[i] >= deltaT_threshold:
        y_Max2 = y[i]
        x_Max2 = i
print("Point over threshold on the right:", x_Max2, y_Max2)

ratio2 = (deltaT_threshold - y_Min2)/(y_Max2 - y_Min2)
x2 = ratio2*(x_Min2 - x_Max2) + x_Max2
print("X-Coordinate right side", x2)

delta_T = x2 - x1
print("delta_T =", delta_T)

m1 = (y_Max-y_Min)/(x_Max-x_Min)
x1_neu = linear(m1, x_Max, y_Max)
m2 = (y_Max2-y_Min2)/(x_Max2-x_Min2)
x2_neu = linear(m2, x_Max2, y_Max2)
delta_T2 = x2_neu - x1_neu
print("New x-Coordinates:", x1_neu, x2_neu)
print("delta_T2 in 0.0:", delta_T2)

#time_sum = 0
#time = [0]*len(y)
#for i in range(len(y)):
    #time[i] = time_sum
    #time_sum = time_sum + T/x_Maximum

plt.figure()
plt.xlim(0,150)
plt.plot(y)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
