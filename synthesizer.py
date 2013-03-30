import sys
import time
import threading
import pyaudio
import numpy as np
import galry.pyplot as plt

# COLORS
# ------
def hue(H):
    H = H.reshape((-1, 1))
    R = np.abs(H * 6 - 3) - 1;
    G = 2 - np.abs(H * 6 - 2);
    B = 2 - np.abs(H * 6 - 4);
    return np.clip(np.hstack((R,G,B)), 0, 1)
    
def hsv_to_rgb(HSV):
    a = HSV[:,1].reshape((-1, 1))
    b = HSV[:,2].reshape((-1, 1))
    a = np.tile(a, (1, 3))
    b = np.tile(b, (1, 3))
    return ((hue(HSV[:,0]) - 1) * a + 1) * b


# SYNTHESIZER
# -----------
class Player(object):
    def __init__(self, rate=None, callback=None):
        self.rate = rate
        self.callback = callback
        self.paused = False

    def _play_chunk(self):
        chunk = self.callback()
        self.stream.write(chunk.astype(np.float32).tostring())

    def _play(self):
        # Open the stream on the background thread.
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.rate, output=1)
        if self.paused:
            self.paused = False
        # Main loop.
        while not self.paused:
            self._play_chunk()
        
    def play(self):
        if not hasattr(self, '_thread'):
            self._thread = threading.Thread(target=self._play)
            self._thread.daemon = True
            self._thread.start()
    
    def pause(self):
        self.paused = True
        time.sleep(.1)
        self.stream.close()
        self._thread.join()
        del self._thread

class Synthesizer(object):
    def __init__(self, rate):
        self.rate = rate
        self.f0 = 523.25
        self.a = 1.
        
    def set_f0(self, f0):
        self.f0 = f0
    
    def set_a(self, a):
        self.a = a
        
    def callback(self):
        x = int(512 * self.f0 / self.rate)
        t = np.arange(0, x * float(self.rate) / self.f0) / float(self.rate)
        return self.a * .75 * np.sin(2 * np.pi * self.f0 * t)

# Create the synthesizer.
rate = 44100
s = Synthesizer(rate)
p = Player(rate=rate, callback=s.callback)

# Create a Figure.
plt.figure(toolbar=False)

# VISUALS
# -------
# Number of discs.
n = 20

# Display n static discs with an opacity gradient.
color = np.ones((n, 4))
color[:,2] = 0
color[:,3] = np.linspace(0.01, 0.1, n)
plt.plot(np.zeros(n), np.zeros(n), 'o', color=color, ms=50, is_static=True)

# Global variable with the current disc positions.
position = np.zeros((n, 2))

# Global variable with the current mouse position.
mouse = np.zeros((1, 2))

# Animation weights for each disc, smaller = slower movement.
w = np.linspace(0.01, 0.1, n).reshape((-1, 1))

# Vertical lines with chromatic and major scales.
line_color = np.ones((50, 4)) * .5
scale = np.array([0, 2, 4, 5, 7, 9, 11])
scale = np.concatenate((scale, scale + 12, [24]))
scale = np.concatenate((2 * scale, 2 * scale + 1))
line_color[scale] = .75
plt.plot(np.repeat(np.linspace(-1., 1., 25), 2),
         np.tile([-1., 1.], 25), 
         color=line_color,
         primitive_type='LINES',
         is_static=True)

plt.xlim(-1.05, 1.05)

# EVENTS
# ------
hsv = np.ones((1, 3))

def update(fig, param, quantize=False):
    # Update the mouse position.
    global mouse
    mouse[0,:] = param['mouse_position']
    x, y = param['mouse_position']
    # Compensate for xlim.
    x = x * 1.05
    # Quantization on chromatic scale if click.
    if quantize:
        x = np.round(x * 12) / 12.
        mouse[0,0] = x / 1.05
    # Change the frequency/volume of the synthesizer.
    s.set_f0(523.25 * 2 ** x)
    s.set_a((y + 1) / 2)
    
    # Set color.
    hsv[:, 0] = (x + 1) % 1
    hsv[:, 1] = (y + 1) / 2
    hsv[:, 2] = (y + 1) / 2
    color[:,:3] = hsv_to_rgb(hsv)
    fig.set_data(color=color)

def mousemove(fig, param):
    update(fig, param, False)

def mouseclick(fig, param):
    update(fig, param, True)
    
# Animate the discs.
def anim(fig, param):
    # The disc position is obtained through a simple linear filter of the
    # mouse position.
    global position
    position += w * (-position + mouse)
    fig.set_data(position=position)
    
# We bind the "Move" action to the "mousemove" callback.
plt.action('LeftClick', mouseclick)
plt.action('LeftClickMove', mousemove)

# We bind the "Animate" event to the "anim" callback.
plt.animate(anim, dt=.01)

# Start the synthesizer.
p.play()

# Show the figure.
plt.show()

# Close the synthesizer at the end.
p.pause()
