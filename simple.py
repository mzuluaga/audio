# check https://realpython.com/python-scipy-fft/BNN
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

def read_wav(f):
  fs, y = scipy.io.wavfile.read(f)
  y = y / float(np.max(np.abs(y)))
  return y, fs

def downsample(y, sr):
  print('original:', len(y), sr)
  if len(y) % 2 != 0:
    y = y[:-1]
  y = (y[::2] + y[1::2]) / 2.0
  sr = sr // 2
  print('new:', len(y), sr)
  return y, sr

def mono(y):
  y = (y[:,0] + y[:,1]) / 2.0
  return y

def sinusoid(A, F, fs=16000, t=1.0):
  ts = np.arange(0.0, t, 1.0 / fs)
  return A * np.sin(2 * np.pi * ts * F)

y, sr = read_wav('audios/SoundFRequenzH3.3.wav')
y = y[1*sr:2*sr]

# sr = 22050
# y = sinusoid(0.5, 440.0, fs=sr, t=1.0) + sinusoid(0.5, 1024.0, fs=sr, t=1.0)
# print(y.shape)

N = len(y)  # N = SAMPLE_RATE * DURATION = 22050 samples * 1 sec
assert N == sr, print(N, sr)

yf = np.fft.rfft(y)
xf = np.fft.rfftfreq(N, 1 / sr)
idx = np.where(np.abs(yf)>200)
print('main frequencies:', xf[idx])


plt.figure(figsize=(20, 6))
plt.plot(y[:512])

plt.figure(figsize=(20, 6))
plt.plot(xf, np.abs(yf))


plt.figure(figsize=(20, 6))
yff = np.abs(yf)
yff[np.abs(yf)<200] = 0.0
plt.plot(xf, np.abs(yff))

yh = np.fft.irfft(yff)

plt.figure(figsize=(20, 6))
plt.plot(yh[:512])

plt.show()
