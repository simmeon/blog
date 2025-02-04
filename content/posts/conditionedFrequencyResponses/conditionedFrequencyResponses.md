---
title: 'Conditioned Frequency Responses'
date: '2025-02-04T19:36:10+13:00'
# weight: 1
# aliases: ["/first"]
tags: ["frequency response", "control", "system identification", "aerospace", "signal processing"]
author: "simmeon"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "Understanding how to find frequency responses when there are multiple correlated inputs present."
canonicalURL: "https://simmeon.github.io/blog/post/"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: false
cover:
    image: "<image path/url>" # image path/url
    alt: "<alt text>" # alt text
    caption: "<text>" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/simmeon/blog/tree/main/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

<!-- Import KaTeX -->
{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}


## An example

Imagine we're looking into a system that has the following sort of form

$$
    \ddot{y}(t) + c \dot{y}(t) + k y(t) = x_{1}(t) + x_{2}(t)
$$

where \\( y \\) is our output and there are two inputs, \\( x_{1} \\) and \\( x_{2} \\). For the purpose 
of this example, we don't know exactly what \\( c \\) and \\( k \\) are - the specific system we have is 
a bit of a mystery. What we can do, however, is measure the inputs we put into the system and 
then measure the output we get.

This is useful because we know that the frequency response of a system is 

$$
    H(f) = \frac{Y(f)}{X(f)}
$$

for some output \\( y \\) and input \\( x \\). We know that we can get lots of useful information 
about how our system behaves from the frequency response - things like where natural frequencies are 
and whether the system will be stable at certain frequencies.

So let's Fourier transform our input and output data and have a look at the frequency responses
we get. And just to compare, let's also plot the analytical frequency responses just to show that we 
did it right. 

{{< figure src="../img/intro-example.png" align=center caption="**Figure 1**: A simple two-input, one-output linear system to introduce the problems that arise with partially correlated inputs." >}}

{{< collapse summary="Figure 1 code" >}}

```python {linenos=true}
'''
A simple two-input (partically correlated), one-output linear system to introduce 
the problems that arise with partially correlated inputs.

Last Modified: 2025-02-04
License: MIT

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, zoom_fft
from scipy.integrate import odeint

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')


pi = np.pi

# ------------------------------------------------------------------------------

# Define and simulate system

# System coefficienets (this is secret don't look...)
a = 2
b = 3
c = 2
d = 0.5
k = (2 * pi * 2)**2 # ~ 158

# Simulation properties
T = 50
fs = 1000
N = T * fs
t = np.linspace(0, T, N, endpoint=False)

# Define input x1: freq sweep
fmin = 0.1
fmax = 5
x1 = chirp(t, fmin, T, fmax)

# Define input x2: partially correlated
x2_uc = np.random.normal(0, 1, N)
x2 = d * x1 + x2_uc

# Define state derivative
def dydt(y, t):
    u1 = x1[int(t * fs) - 1]
    u2 = x2[int(t * fs) - 1]
    return np.array([y[1], - k * y[0] - c * y[1] + a * u1 + b * u2])
    
# Simulate system
y0 = [0, 0]
y = odeint(dydt, y0, t)
y = y[:,0]

# ------------------------------------------------------------------------------

# Frequency Responses

# Analytical frequency response
f = np.logspace(np.log10(fmin), np.log10(fmax), N)
s = 1.0j * 2 * pi * f

H1y = a / (s**2 + c * s + k)
H2y = b / (s**2 + c * s + k)

# Estimated response
Y = zoom_fft(y, [fmin, fmax], m=N, fs=fs)
X1 = zoom_fft(x1, [fmin, fmax], m=N, fs=fs)
X2 = zoom_fft(x2, [fmin, fmax], m=N, fs=fs)

H1y_est = Y / X1
H2y_est = Y / X2

f_est = np.linspace(fmin, fmax, N)

# ------------------------------------------------------------------------------

# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,3)
ax3 = fig.add_subplot(2,2,2)
ax4 = fig.add_subplot(2,2,4)


ax1.set_title(r'$Y / X_{1}$ Magnitude')
ax1.semilogx(f, 20 * np.log10(abs(H1y)), label='Analytical')
ax1.semilogx(f_est, 20 * np.log10(abs(H1y_est)), label='Estimated')

ax2.set_title(r'$Y / X_{1}$ Phase')
ax2.semilogx(f, np.rad2deg(np.angle(H1y)), label='Analytical')
ax2.semilogx(f_est, np.rad2deg(np.angle(H1y_est)), label='Estimated')

ax3.set_title(r'$Y / X_{2}$ Magnitude')
ax3.semilogx(f, 20 * np.log10(abs(H2y)), label='Analytical')
ax3.semilogx(f_est, 20 * np.log10(abs(H2y_est)), label='Estimated')

ax4.set_title(r'$Y / X_{2}$ Phase')
ax4.semilogx(f, np.rad2deg(np.angle(H2y)), label='Analytical')
ax4.semilogx(f_est, np.rad2deg(np.angle(H2y_est)), label='Estimated')

ax1.set_ylabel('Magnitude [dB]')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Phase [deg]')
ax4.set_xlabel('Frequency [Hz]')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()
```

{{< /collapse >}}

Woah okay that's not great...

The noise is expected due to what our inputs are and how the Fourier transform and frequency response is computed, but the magnitude of both our calculated frequency responses is definitely much higher than it should be. 

What's going on here? As a hint of where this is going, it has to do with our two inputs being partially
correlated. In this case that means that \\( x_{2} \\) is partially made up of \\( x_{1} \\).


## What is this all about?

A frequency response gives information about how an output of a system will react to a sinusoidal input at different frequencies. They are very useful in analysing the stability and behaviour of systems. They are also central to frequency-domain system identification methods, which is what we will be interested in using them for. 

As we already said, for some system input \\( x(t) \\) and output \\( y(t) \\), the frequency response is

$$
    H(f) = \frac{Y(f)}{X(f)}
$$

**But this is not always entirely true**. In systems where there are *multiple partially correlated inputs* and those inputs all affect the output, we have to be more careful about how we calculate these frequency responses. In other words, we have to *condition* the frequency responses to remove the effects that 
come from this correlation.

Throughout this post, we will develop a very useful way to calculate frequency responses (especially for system identification) using spectral density functions. Then, we will apply this method in single-input single-output (SISO) systems. We will build up some theory on how to describe these correlation effects in multiple-input single-output (MISO) systems and see why the above equation for frequency responses is lacking. Finally, we will discuss how to change our method to *condition* these frequency responses so 
they are correct.


## Stationary Random Processes

We are mostly going to be interested in frequency responses for the purpose of system identification. That means we will have real measured data for \\( x(t) \\) and \\( y(t) \\). If we then want to find the frequency response, you may first think to just follow the earlier equation and Fourier transform both of these.

And that would work! But it's important to remember that we're working with real-world measured data here, and real-world data always has a lot of noise, disturbances, and other issues that make our clean theoretical equations less useful. Take the following example in **Figure 1**:

{{< figure src="../img/freq-response-straight-fourier.png" align=center caption="**Figure 1**: Seeing how noisy data can affect direct calculation of the frequency response from Fourier transforms." >}}

{{< collapse summary="Figure 1 code" >}}

```python {linenos=true}
'''
Seeing how noisy data can affect direct calculation of the frequency
response from Fourier transforms.

Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.fft import rfft, rfftfreq


plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

pi = np.pi

# Let's set up a simple spring, mass, damper system to mess around with.
fn = 1              # natural frequency of system [Hz]
zeta = 0.05          # system damping ratio

wn = 2 * pi * fn    # natural freq [rad/s]

# We can find the analytical transfer function to get the true frequency response
def H_func(f):
    s = 1.0j * 2 * pi * f
    return wn**2 / (s**2 + 2 * zeta * wn * s + wn**2)

fmin = 0.1  # min frequency in response
fmax = 10   # max frequency in response
f_analytical = np.logspace(np.log10(fmin), np.log10(fmax), 500) # array of freqs to calculate response for

H_analytical = H_func(f_analytical)                     # Frequency response
mag_analytical = 20 * np.log10(abs(H_analytical))       # Magnitude of response (in dB)
phase_analytical = np.rad2deg(np.angle(H_analytical))   # Phase of response (in deg)


# Now let's create some noisy input and output data to simulate measurements
# To do this we will solve the system with a frequency sweep over the range of interest (0.1 - 10 Hz)

T = 20  # period to simulate system over [s]

# Define our input function, a frequency sweep
# NOTE: to get a correct frequency response over all frequencies of interest, the input must
# necessarily contaion information on all the frequencies you are interested in. 
def u_func(t):
    f = t / T * fmax * 2
    return 10 * np.sin(2 * pi * f * t)

# Define the derivative function of the system (state space derivative)
def dxdt(x, t):
    u = u_func(t)
    return np.array([x[1], - wn**2 * x[0] - 2 * zeta * wn * x[1] + wn**2 * u])


# Solve system
N = 5000                   # number of sample points
t = np.linspace(0, T, N)    # time array
x0 = np.array([0, 0])       # initial state conditions
sol = odeint(dxdt, x0, t)

y = sol[:, 0]   # pull out position from state as our output
x = u_func(t)   # get array of inputs for use later

# Now we will add some output noise to simulate noisy measurements
y = y + np.random.normal(0, 2, len(y))

# Then, we can fourier transform our input and output to calculate the freqency response
Y = rfft(y)
X = rfft(x)
f_estimated = rfftfreq(N, T / N)
H_estimated = Y / X                                   # Estimated freq response from noisy data
mag_estimated = 20 * np.log10(abs(H_estimated))       # Magnitude of response (in dB)
phase_estimated = np.rad2deg(np.angle(H_estimated))   # Phase of response (in deg)


# Plotting
plt.figure(figsize=(12, 8))

# Subplot 1: Time domain response
plt.subplot(3, 1, 1)
plt.plot(t, y)
plt.title("System Response (y) over Time", fontsize=20)
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Output (y)", fontsize=18)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=16)

# Subplot 2: Bode magnitude plot
plt.subplot(3, 1, 2)
plt.semilogx(f_estimated, mag_estimated, label="Estimated", linewidth=2)
plt.semilogx(f_analytical, mag_analytical, label="Analytical", linestyle='--', linewidth=2)
plt.title("Bode Magnitude Plot", fontsize=20)
plt.ylabel("Magnitude (dB)", fontsize=18)
plt.legend(fontsize=16)
plt.grid(True, which='both', axis='both')
plt.xscale('log')
plt.xlim((fmin, fmax))
plt.tick_params(axis='both', which='major', labelsize=12)

# Subplot 3: Bode phase plot
plt.subplot(3, 1, 3)
plt.semilogx(f_estimated, phase_estimated, label="Estimated", linewidth=2)
plt.semilogx(f_analytical, phase_analytical, label="Analytical", linestyle='--', linewidth=2)
plt.title("Bode Phase Plot", fontsize=20)
plt.ylabel("Phase (Degrees)", fontsize=18)
plt.xlabel("Frequency (Hz)", fontsize=18)
plt.legend(fontsize=16)
plt.grid(True, which='both', axis='both')
plt.xscale('log')
plt.xlim((fmin, fmax))
plt.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()
```

{{< /collapse >}}

While the frequency response estimate is still pretty good where it matters, you can see how the noise starts to introduce larger errors as the frequency increases. What would be nice is a way of somehow modelling this noise and creating methods that are able to account for noise and let us minimise the impact of it.

To do this we will turn to some statistical methods and see where they can take us...

### Basics

First, let's look at the concept of a stationary random process. A random process meaning some function whose value at any given time is random. Each time index can be described by a random variable. This is good as our measurement noise can be modelled by these random variables. The random process is *stationary* if the probability density function of the random variable at any time is the same.

Let's look at an example to help. In **Figure 2**, we are using random normal noise as our random variable at each time \\( t_{i} \\) to generate a '*sample function*', \\( x_{1}(t) \\), of the random process. 

If we do this multiple times to create lots of sample functions \\( x_{k}(t), k = 1, 2, ... \\), we can then estimate the probability density function of the random variable at a particular time \\( t_{i} \\). For example, if we focus on time \\( t = 0 \\), we could estimate the probability density function \\( f_{X}(x) \\) by counting how many sample functions had a value \\( x_{k}(0) \\) within certain ranges. Eg. how many had a value between 0 and 0.5? Between 0.5 and 1? Between -2 and -1.5?

Basically we are creating a histogram of all the \\( x_{k}(0) \\) values. We can then do the same thing for every time index \\( t_{i} \\).


{{< figure src="../img/stationary-random-process.png" align=center caption="**Figure 2**: Exploring what a stationary random process is and some useful features of it." >}}

{{< collapse summary="Figure 2 code" >}}

```python {linenos=true}
'''
Exploring what a stationary random process is and some useful features of it.
Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import histogram

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# We will want to store each 'sample function' in an array
x = []

N = 50 # sample length
T = 10 # time period sample is over
t = np.linspace(0, T, N)
num_samples = 100 # number of sample functions to run

for i in range(num_samples):
    # The stationary random process we will use is random normal noise
    x.append(np.random.normal(0, 1, N))


# We can then estimate the probability density function at a particular time (t)
# by counting how many of the sample functions have values within a particular 
# range at that time.
ti_data = np.zeros((N, num_samples))
pdfs = []
num_bins = 20
pdf_vals = np.linspace(-5, 5, num_bins)
for i in range(N):
    for j in range(num_samples):
        ti_data[i, j] = (x[j][i])
    pdfs.append(histogram(ti_data[i], -5, 5, num_bins) / num_samples)


# Plotting
fig= plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(2, 2, 2)
ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title("One Sample Function", fontsize=20)
ax1.set_xlabel("t", fontsize=20)
ax1.set_ylabel("x(t)", fontsize=20)
for i in range(1):
    ax1.plot(t, x[i], alpha=1.0)

ax1.set_ylim((-4, 4))


ax2.set_title(r"Probability Density Function for $t = 0$", fontsize=20)
ax2.set_xlabel(r"$x(t = 0$)", fontsize=20)
ax2.set_ylabel(r"$f_{X}(x)$", fontsize=20)
for i in range(1):
    ax2.plot(pdf_vals, pdfs[i], alpha=1.0)

ax2.set_ylim((0, 0.3))


ax3.set_title("All Sample Functions", fontsize=20)
ax3.set_xlabel("t", fontsize=20)
ax3.set_ylabel("x(t)", fontsize=20)
for i in range(num_samples):
    ax3.plot(t, x[i], alpha=0.05)

ax3.plot(t, x[0], color='yellow')

ax3.set_ylim((-4, 4))


ax4.set_title(r"Probability Density Functions for each $t_{i}$", fontsize=20)
ax4.set_xlabel(r"$x(t_{i}$)", fontsize=20)
ax4.set_ylabel(r"$f_{X}(x)$", fontsize=20)
for i in range(N):
    ax4.plot(pdf_vals, pdfs[i], alpha=0.1)

ax4.plot(pdf_vals, pdfs[0], alpha=0.1)

ax4.set_ylim((0, 0.3))

plt.show()
```

{{< /collapse >}}

Something that stands out is how the probability density function for every time index looks about the same. And, in fact, if we had worked it out analytically instead of estimating it, we would see that it is exactly the same probability density function for every time. This is the key feature that makes our random process a *stationary* random process.

This is really useful because it allows us to model a function over time in a way that doesn't care about what specific time we are looking at. All the information about what value the function may be at any given time is encapsulated by a single probability density function that's independant of time.

But why is this going to be useful? What can we do with this? And how does it relate to calculating frequency responses?

You might be starting to think that if we assume our input and output records are stationary random processes, we might be able to do some fancy statistical manipulation with them. Which is exactly what we will try do next...

### Correlation Functions

What kinds of things can we do with these stationary random processes? Let's use a different, simpler example to help.

Take the following digital signal which can either be 1 V or 0 V to represent either a 1 or 0 bit. Each bit has a length of \\( T \\) and each sample function may be offset from others. We will also assume it is random whether we see a 1 V or 0 V at any time. **Figure 3** has some example sample functions of this signal.

{{< figure src="../img/digital-signal-srp.png" align=center caption="**Figure 3**: Some sample functions of our digital signal stationary random process." >}}

{{< collapse summary="Figure 3 code" >}}

```python {linenos=true}
'''
Some sample functions of our digital signal stationary random process.
Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

num_bits = 10

# Calculate random series of bits
bits1 = np.random.randint(0, 2, num_bits)
bits2 = np.random.randint(0, 2, num_bits)
bits3 = np.random.randint(0, 2, num_bits)

N = 100 # Number of sample points per bit

# Make length of each bit 1 s
t = np.linspace(0, num_bits, num_bits * N)

# sample functions
x1 = []
x2 = []
x3 = []

# Make sample function have N points per bit and add offsets
for bit in bits1:
    x1 = x1 + [bit] * N

for bit in bits2:
    x2 = x2 + [bit] * N
x2 = x2[35:] + x2[:35] # offset

for bit in bits3:
    x3 = x3 + [bit] * N
x3 = x3[60:] + x3[:60] # offset


# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

ax1.set_title('Digital Signal Sample Functions', fontsize=20)

ax1.set_xticks(np.arange(0, 11, 1))
ax2.set_xticks(np.arange(0, 11, 1))
ax3.set_xticks(np.arange(0, 11, 1))

ax1.grid()
ax2.grid()
ax3.grid()

ax1.grid(visible=True, axis='x', which='both')
ax2.grid(visible=True, axis='x', which='both')
ax3.grid(visible=True, axis='x', which='both')

ax1.plot(t, x1)
ax2.plot(t, x2)
ax3.plot(t, x3)

ax2.set_ylabel('Voltage (V)', fontsize=20)
ax3.set_xlabel('Time (s)', fontsize=20)

plt.show()
```

{{< /collapse >}}

Hopefully you can convince yourself that the probability density function of this random process will be the same at any time: in half the samples we will get a 1 and the other half we will get a 0. So this is another example of a stationary random process.

One fairly simple thing we could look at is the mean (or expected value) of this stationary random process. In more formal terms, we would use the expectation operator:

$$
    \text{mean} = E\\{x_{k}(t)\\}
$$

Since the probability density function is the same for all time, we can also think of the expected value as the integral of the probability density function over all possible values the random variable can be. Or, more simply, the area under the probability density function.

$$
    E\\{x(t)\\} = \int_{-\infty}^{\infty}{x f_{X}(x)} dx
$$

In our digital signal example, it's pretty simple to calculate this expected value since we only have two discrete values the random variable can be: 0 and 1. And each has a 50 % probability, giving

$$
    \text{mean} = 0 \times 0.5 + 1 \times 0.5 = 0.5 \text{ V}
$$

Building on this idea, we might decide to put some other quantity in this expectation operator. For example, 

$$
    E\\{x(t) x(t + \tau) \\}
$$

where \\( \tau \\) here is acting as a time offset. Now we know that since this is a stationary random process, the probabilty density function is time independant, so the only variable in this new function is \\( \tau \\). 

Let's name this new function the *correlation function*, for reasons we should see soon. 

$$
    R_{xx}(\tau) = E\\{x(t) x(t + \tau) \\}
$$

Calculating some examples should help see what this correlation function tells us. Let's start with the simplest case where \\( \tau = 0 \\). That means we will be calculating the expected value

$$
    E\\{x(t) x(t) \\}
$$

where again, since this our probability density function is independant of time, means we can choose any time to evaluate this at. Referring back to **Figure 3** let's look at the purple line where \\( t = 7 \\). 

The value of the random variable \\( x(t = 7) \\) will be 1 half the time and 0 half the time. So when multiplying it with itself and taking the expected value we will get:

$$
    R_{xx}(\tau = 0) = (1 \times 1) \times 0.5 + (0 \times 0) \times 0.5 = 1 \text{ V}^{2}
$$

Things get more interesting when \\( \tau \\) gets larger - specifically larger than the bit length \\( T \\). Let's use \\( x(t = 7) \\) again but this time have \\( \tau = 1.8 \\), which is shown on **Figure 3** as the orange line.

In this case, \\( x(7) \\) will be 1 half the time and 0 half the time as before, but now, \\( x(7 + 1.8) \\) will also be 1 half the time and 0 half the time, independant of what \\( x(7) \\) is. This gives us four possible combinations:

$$
    R_{xx}(\tau = 1.8) = (1 \times 1 ) + (1 \times 0) \times 0.5 + (0 \times 1) \times 0.5 + (0 \times 0) \times 0.5
$$

$$
    R_{xx}(\tau = 0) = 0.25 \text{ V}^{2}
$$

As a final case, let's look at a value of \\( \tau \\) that is less than the bit period \\( T \\). For example, \\( \tau = 0.5 \\). As before, \\( x(7) \\) will be 1 half the time and 0 half the time, but calculating the probability that \\( x(7 + 0.5) \\) will be 1 or 0 is slightly harder due to the offset we have included. Notice in **Figure 3** that if \\( x(7) = 0 \\), it's somewhat more likely that \\( x(7 + 0.5) \\) will also be a 0 than a 1, because they might fall in the same bit (and vice versa if \\( x(7) = 1 \\)). In fact, the smaller \\( \tau \\) is, the higher the probability that the two random variables will have the same value, affecting our correlation function.

We can plot the full correlation function over a range of \\( \tau \\) values as shown in **Figure 4**.

{{< figure src="../img/digital-signal-correlation-function.png" align=center caption="**Figure 4**: Plotting the correlation function of the digital signal stationary random process example." >}}

{{< collapse summary="Figure 4 code" >}}

```python {linenos=true}
'''
Plotting the correlation function of the digital signal stationary random process example.
Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Array of tau values to calculate the correlation function over
tau = np.linspace(-5, 5, 500)

# Correlation function
# NOTE: this is not calculated properly in any way,
# it is quick and dirty but gives the correct result
def Rxx(tau_array):
    Rxx = []
    for tau in tau_array:
        if abs(tau) < 1:
            Rxx.append(((1 - abs(tau)) / 1) * 0.25 + 0.25)
        else:
            Rxx.append(0.25)
    
    return Rxx

R = Rxx(tau)

plt.plot(tau, R)

plt.title('Correlation Function of Digital Signal', fontsize=20)
plt.xlabel(r'$\tau$', fontsize=20)
plt.ylabel(r'$R_{xx}(\tau)$', fontsize=20)
plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xticks(np.arange(-5, 6, 1))
plt.ylim((0, 1))

plt.show()
```

{{< /collapse >}}

As we can see, as \\( \tau \\) gets smaller than the bit length \\( T \\), the correlation function increases linearly. When \\( \tau \\) is bigger than \\( T \\), the two random variables are completely independant and have the value of \\( 0.25 \text{ V}^{2} \\) as we calculated. And when \\( \tau = 0 \\) we get \\( 0.5 \text{ V}^{2} \\) -- also what we calculated.

You may be starting to realise now why this is called the correlation function. It is telling us something about whether we can predict the future value of a function based on its current value. In other words, it tells us whether there is any *correlation* between the current value and the future value. 

When \\( \tau = 0 \\), that future value is always the current value, so we have the highest correlation. As \\( \tau \\) gets bigger, we get less sure about whether we are still on the same bit and therefore if we are able to predict the future value. Finally, after \\( \tau > 1\\), we are definitely on a different bit and so have no way of predicting the future value from our current value.

Note, however, that when \\( \tau > 1 \\) the correlation function is *not* 0. This is somewhat deceptive as it suggests there is some correlation even though we know this isn't true. Because of this, a better name for this function may be the 'average shared directional power' [1]. This won't be very relevant for how we will be using this function, but it's worth keeping in mind.

To summarise, we now have a tool called the correlation function defined as 

$$
    R_{xx}(\tau) = E\\{x(t) x(t + \tau) \\}
$$

When we use the same stationary random process, we call this the auto-correlation function. We can also use two different stationary random processes:

$$
    R_{xy}(\tau) = E\\{x(t) y(t + \tau) \\}
$$

which we would call the cross-correlation function. This can be interpreted as telling us about whether a future value of \\( y(t) \\) can be predicted from the current value of \\( x(t) \\).

While it may not seem like it yet, these correlation functions form the basis for the method we will use to derive our frequency responses. But we need to derive one more function before we get there.

### Spectral Density Functions

We know that, to get our frequency responses, we will have to go into the frequency domain at some point. Spectral density functions will give our statistical methods derived so far a link between the time and frequency domains.

#### Spectra Via Correlation Function

Transforming our newly-derived correlation function into the frequency domain seems like a good place to start. Luckily, this is pretty straightforward with the Fourier transform.

$$
    S_{xx}(f) = \int_{-\infty}^{\infty}{R_{xx}(\tau) e^{-j 2 \pi f \tau}} d\tau
$$

As this integral is from \\( -\infty \\) to \\( \infty \\), we will get both positive and negative frequencies as a result. \\( S_{xx}(f) \\) is the *two-sided power spectral density* function. It related to *power* since the correlation function has an \\( x^{2} \\) relationship, often proportional to power. It is a *spectral density* because the Fourier transform gives us power per Hertz, ie. a density over frequency.

When dealing with real systems, we don't really care about negative frequencies. And since the Fourier transform is symmetric, we don't get any additional information from those negative frequencies. So we can instead use the *one-sided* spectral density function \\( G_{xx}(f) \\) where we cut off the negative frequencies and shift all that power into the positive frequencies:

$$
    G_{xx}(f) = 2 S_{xx}(f) \text{ , where } f > 0
$$

Much like with the correlation function, if we use the same two stationary random processes, we will call this the autospectral density function.

We can also define the cross spectral density function with two different processes:

$$
    G_{xy}(f) = 2 S_{xy}(f) \text{ , where } f > 0
$$

**Figure 5** shows both the one-sided and two-sided power spectral density functions for our digital signal example.

{{< figure src="../img/digital-signal-spectral-density.png" align=center caption="**Figure 5**: Plotting the spectral density function of the digital signal stationary random process example." >}}

{{< collapse summary="Figure 5 code" >}}

```python {linenos=true}
'''
Plotting the spectral density function of the digital signal stationary random process example.
Created by: simmeon
Last Modified: 2025-01-25
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import zoom_fft

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Array of tau values to calculate the correlation function over
n = 5000
T = 5

tau = np.linspace(-T/2, T/2, n)

# Correlation function
# NOTE: this is not calculated properly in any way,
# it is quick and dirty but gives the correct result
def Rxx(tau_array):
    Rxx = []
    for tau in tau_array:
        if abs(tau) < 1:
            Rxx.append(((1 - abs(tau)) / 1) * 0.25 + 0.25)
        else:
            Rxx.append(0.25)
    
    return Rxx

R = Rxx(tau)

# Fourier transform of correlation function using a fancy chirp-z tranform method
# for better frequency resolution
G = zoom_fft(R, [0, 1], m=len(R)//2, fs=n/(2*T))
G = abs(G) / n

# Frequencies corresponding to Fourier transform data
f = np.linspace(0, 1, len(R)//2)

# zoom_fft gives only positive freqs, aka. one-sided
# since Fourier is symmetric, we can flip it for negative freqs
S_negative = np.flip(G)
f_negative = - np.flip(f)

G = G.tolist()
S_negative = S_negative.tolist()
f = f.tolist()
f_negative = f_negative.tolist()

# Adding positive and negative freqs and halving amplitude gives full two-sided spectral density
S = S_negative + G
f = f_negative + f
S = np.array(S)
S = S / 2

# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(f[n//2:], S[n//2:], color='white')
ax1.plot(f[0:n//2], S[0:n//2], color='white')

ax2.plot(f[n//2:], G, color='white')

ax1.set_title('Two and One-sided Spectral Density Functions of Digital Signal', fontsize=20)
ax2.set_xlabel('f', fontsize=20)
ax1.set_ylabel(r'$S_{xx}(\tau)$', fontsize=20)
ax2.set_ylabel(r'$G_{xx}(\tau)$', fontsize=20)
ax1.set_xlim((-1, 1))
ax2.set_xlim((-1, 1))
ax1.set_ylim((0, 0.3))
ax2.set_ylim((0, 0.3))
ax1.set_xticks(np.arange(-1, 1.1, 0.1))
ax2.set_xticks(np.arange(-1, 1.1, 0.1))

plt.show()
```

{{< /collapse >}}

This is the final tool we need to calculate our frequency responses. However, calculating these spectral density functions is quite a process. In particular, calculating or estimating the probability density functions of our stationary random processes can be very tricky. 

It would be nice if there was a simpler way to get these spectral densities...

#### Spectra Via Finite Fourier Transform

Luckily, there is an easier way! Unfortunately, it's a bit messy to derive.

We will start by showing the result we want to end up with, then showing that it gives the same result as the previous method.

We will be deriving the cross-spectra of two different stationary random processes, \\( x(t) \\) and \\( y(t) \\). And what we are going to guess is that for a *finite* time interval \\( 0 < t < T \\), our cross-spectra will be

$$
    S_{xy}(f, T, k) = \frac{1}{T} X_{k}^{*}(f, T) Y_{k}(f, T)
$$

where \\( X_{k}^{*}(f, T) \\) is the complex conjugate of the finite Fourier transform of the stationary  random process \\( x_{k}(t) \\) over the interval \\( [0, T] \\). Similarly, \\( Y_{k}(f, T) \\) is the finite Fourier transform of \\( y_{k}(t) \\) over the same interval.

$$
    X_{k}(f, T) = \int_{0}^{T}{x_{k}(t) e^{-j 2 \pi f t}}dt
$$

$$
    Y_{k}(f, T) = \int_{0}^{T}{y_{k}(t) e^{-j 2 \pi f t}}dt
$$

This would be a very useful result that would make calculating spectral density functions much easier. What we need to prove is that our guess will lead to the same definition as earlier:

$$
    S_{xy}(f) = \int_{-\infty}^{\infty}{R_{xy}(\tau) e^{-j 2 \pi f \tau}} d\tau
$$

First of all, our guess is currently dependant on a finite value of \\( T \\) and is defined in terms of the stationary random processes. To turn in into something more useful, we need to take the limit as \\( T \to \infty \\) and then also take the expected value of the stationary random processes.

$$
    S_{xy}(f) = \lim_{T \to \infty} E\\{ S_{xy}(f, T, k) \\}
$$

Let's now write out fully what \\( S_{xy}(f, T, k) \\) looks like.

$$
    S_{xy}(f, T, k) = \frac{1}{T} \int_{0}^{T}{x_{k}(\alpha) e^{-j 2 \pi f \alpha}}d\alpha \int_{0}^{T}{y_{k}(\beta) e^{-j 2 \pi f \beta}}d\beta
$$
 
Instead of \\( t \\) we will use \\( \alpha \\) and \\( \beta \\) to help make it clear which variables are in which integral.

$$
    S_{xy}(f, T, k) = \frac{1}{T} \int_{0}^{T} \int_{0}^{T} x_{k}(\alpha) y_{k}(\beta) e^{-j 2 \pi f ( \beta -\alpha)} d\alpha d\beta
$$

We also know that, if we want this to match our earlier derivation, this integral will have to be over \\( \tau \\). So we will change our integration variables to be \\( \alpha \\) and \\( \tau \\), where \\( \tau = \beta - \alpha \\). This should make sense as \\( \tau \\) was the difference between the two times we would look at when creating the correlation function.

As for how this will change our integral, we can show that

$$
    \int_{0}^{T} \int_{0}^{T} d\alpha d\beta = \int_{-T}^{0} \int_{-\tau}^{T} d\alpha d\tau + \int_{0}^{T} \int_{0}^{T - \tau} d\alpha d\tau
$$

where both sides give the value \\( T^{2} \\).

Putting our full integral back in gives us

$$
    S_{xy}(f, T, k) = \int_{-T}^{0} \bigg[ \frac{1}{T} \int_{-\tau}^{T} x_{k}(\alpha) y_{k}(\alpha + \tau) d\alpha \bigg] e^{-j 2 \pi f \tau} d\tau
    + \int_{0}^{T} \bigg[ \frac{1}{T} \int_{0}^{T - \tau} x_{k}(\alpha) y_{k}(\alpha + \tau) d\alpha \bigg] e^{-j 2 \pi f \tau} d\tau
$$

We can then take the expected value of both sides, remembering that \\( R_{xy}(\tau) = E\\{ x_{k}(\alpha) y_{k}(\alpha + \tau) \\} \\).

$$
    E\\{ S_{xy}(f, T, k) \\} = \int_{-T}^{0} \bigg[ \frac{1}{T} \int_{-\tau}^{T} R_{xy}(\tau) d\alpha \bigg] e^{-j 2 \pi f \tau} d\tau
    + \int_{0}^{T} \bigg[ \frac{1}{T} \int_{0}^{T - \tau} R_{xy}(\tau) d\alpha \bigg] e^{-j 2 \pi f \tau} d\tau
$$

If we look at the integrals inside the square brackets, we can actually evaluate these fairly easily.

$$
\begin{aligned}
    \frac{1}{T} \int_{-\tau}^{T} R_{xy}(\tau) d\alpha &= R_{xy}(\tau) \frac{1}{T} \int_{-\tau}^{T} d\alpha \\\ 
    &= R_{xy}(\tau) \frac{1}{T} \big[ \alpha \big]_{\alpha = -\tau}^{T}
    \\\
    &= R\_{xy}(\tau) \frac{1}{T} (T + \tau)
    \\\
    &= R\_{xy}(\tau) (1 + \frac{\tau}{T})
\end{aligned}
$$

$$
\begin{aligned}
    \frac{1}{T} \int_{0}^{T - \tau} R_{xy}(\tau) d\alpha &= R_{xy}(\tau) \frac{1}{T} \int_{0}^{T - \tau} d\alpha \\\ 
    &= R_{xy}(\tau) \frac{1}{T} \big[ \alpha \big]_{\alpha = 0}^{T - \tau}
    \\\
    &= R\_{xy}(\tau) \frac{1}{T} (T - \tau)
    \\\
    &= R\_{xy}(\tau) (1 - \frac{\tau}{T})
\end{aligned}
$$

Subbing these back into our equation gives

$$
    E\\{ S_{xy}(f, T, k) \\} = \int_{-T}^{0} R\_{xy}(\tau) (1 + \frac{\tau}{T}) e^{-j 2 \pi f \tau} d\tau
    + \int_{0}^{T} R\_{xy}(\tau) (1 - \frac{\tau}{T}) e^{-j 2 \pi f \tau} d\tau
$$

Let's now take the limit as \\( T \to \infty \\) of both sides. This will make \\( \frac{\tau}{T} \to 0 \\).

$$
    \lim_{T \to \infty} E\\{ S_{xy}(f, T, k) \\} = \int_{-\infty}^{0} R\_{xy}(\tau) e^{-j 2 \pi f \tau} d\tau
    + \int_{0}^{\infty} R\_{xy}(\tau) e^{-j 2 \pi f \tau} d\tau
$$

We now have the same integrand in both integrals and the upper bound of the first is the same as the lower bound of the second (they are both 0). This means we can combine them into a single integral giving

$$
    \lim_{T \to \infty} E\\{ S_{xy}(f, T, k) \\} = \int_{-\infty}^{\infty} R\_{xy}(\tau) e^{-j 2 \pi f \tau} d\tau = S_{xy}(f)
$$

Great! This shows that our guess from the beginning actually does give the same spectral density function. To be clear, we can write

$$
    S_{xy}(f) = \lim_{T \to \infty} \frac{1}{T} E\\{ X_{k}^{*}(f, T) Y_{k}(f, T) \\}
$$

and similarly with our one-sided spectral density functions

$$
    G_{xy}(f) = \lim_{T \to \infty} \frac{2}{T} E\\{ X_{k}^{*}(f, T) Y_{k}(f, T) \\} \text{ , where } f > 0
$$

In practice, \\( T \\) will always be a finite value, and we will not be able to calculate the true expected value of the Fourier transforms of our stationary random processes. Instead, we will roughly estimate that the expected value will be what our single sample functions give us: \\( x(t) \\) and \\( y(t) \\). This gives us a rough spectral density estimate of

$$
    \hat{G}_{xy}(f) = \frac{2}{T} \big[ X^{*}(f) Y(f) \big]
$$

where \\( X(f) \\) and \\( Y(f) \\) are the finite Fourier transforms of our single sample functions.

We can see what this estimate looks like in **Figure 6** and **Figure 7**.

{{< figure src="../img/digital-signal-spectral-density-comparison.png" align=center caption="**Figure 6**: Plotting the spectral density and estimated spectral density functions of the digital signal stationary random process example." >}}

{{< figure src="../img/multiple-estimates-spectral-density.png" align=center caption="**Figure 7**: Plotting the spectral density and 100 estimated spectral density functions." >}}

{{< collapse summary="Figure 6 & 7 code" >}}

```python {linenos=true}
'''
Estimated spectral density function of the digital signal from direct finite Fourier transforms.
Created by: simmeon
Last Modified: 2025-01-26
License: MIT

'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import zoom_fft

plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

# Same proper G from last example
# Array of tau values to calculate the correlation function over
n = 5000
T = 5

tau = np.linspace(-T/2, T/2, n)

# Correlation function
# NOTE: this is not calculated properly in any way,
# it is quick and dirty but gives the correct result
def Rxx(tau_array):
    Rxx = []
    for tau in tau_array:
        if abs(tau) < 1:
            Rxx.append(((1 - abs(tau)) / 1) * 0.25 + 0.25)
        else:
            Rxx.append(0.25)
    
    return Rxx

R = Rxx(tau)

# Fourier transform of correlation function using a fancy chirp-z tranform method
# for better frequency resolution
G = zoom_fft(R, [0, 1], m=len(R)//2, fs=n/(2*T))
G_actual = abs(G) / len(R)
f_actual = np.linspace(0, 1, len(R)//2)

# --------------------------------------------------------------------- #
G_array = []
for i in range(100):
    num_bits = 10

    # Calculate random series of bits
    bits = np.random.randint(0, 2, num_bits)

    N = 100 # Number of sample points per bit
    bit_length = 1 # seconds

    # Make length of each bit 1 s
    T = num_bits * bit_length
    t = np.linspace(0, T, num_bits * N)

    # sample function
    x = []

    # Make sample function have N points per bit and add offsets
    for bit in bits:
        x = x + [bit] * N

    # Fourier transform sample function
    X = zoom_fft(x, [0, 1], m=len(x)//2, fs=N)
    X = abs(X) / len(x) * 2
 
    # Calculate estimated spectral density
    G = 2 / T * (np.conjugate(X) * X)
    G_array.append(G)


# Frequencies corresponding to Fourier transform data
f = np.linspace(0, 1, len(x)//2)

for G in G_array:
    plt.plot(f, G, alpha=0.05)

plt.plot(f_actual, G_actual, label='Actual')

plt.title('100 Estimates of Spectral Density Function', fontsize=20)
plt.xlim((0, 1))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.ylabel(r'$G_{xx}(f)$', fontsize=20)
plt.xlabel('f', fontsize=20)
plt.legend(fontsize=20)

plt.show()
```

{{< /collapse >}}

Now that we have a good way of calculating these spectra, let's find out how they will be useful for finding frequency responses...

## Single Input Single Output (SISO) Frequency Responses

Starting from the familiar convolution integral that describes a SISO system,

$$
    y(t) = \int_{0}^{\infty} h(\alpha) x(t - \alpha) d\alpha
$$

we can also describe the system at time \\( t + \tau \\) as 

$$
    y(t + \tau) = \int_{0}^{\infty} h(\alpha) x(t + \tau - \alpha) d\alpha
$$

Multiplying both sides by \\( x(t) \\) gives

$$
    x(t) y(t + \tau) = \int_{0}^{\infty} h(\alpha) x(t) x(t + \tau - \alpha) d\alpha
$$

Remember that we are going to assume our input \\( x(t) \\) and output \\( y(t) \\) are both stationary random processes, so we can take the expected value of both sides to get

$$
    E\\{x(t) y(t + \tau)\\} = \int_{0}^{\infty} h(\alpha) E\\{x(t) x(t + \tau - \alpha)\\} d\alpha
$$

$$
    R_{xy}(\tau) = \int_{0}^{\infty} h(\alpha) R_{xx}(\tau - \alpha) d\alpha
$$

Now we have the same form of convolution integral just in terms of correlation functions instead of our stationary random processes. So we can transform this into the Fourier domain to change the convolution integral into a multiplication.

$$
    S_{xy}(f) = H(f) S_{xx}(f)
$$

Finally getting the relationship between our statistical methods and the frequency response we've been building up.

$$
    H(f) = \frac{S_{xy}(f)}{S_{xx}(f)} = \frac{G_{xy}(f)}{G_{xx}(f)}
$$


## Multiple Input Single Output (MISO) Frequency Responses

### Theory on Partially Correlated Inputs

To be able to understand what happens when we have correlated inputs, we first need to think about what a system with correlated inputs would look like. To do this, we will make use of some helpful system diagrams.

**Figure 8** shows a diagram of a single-input, single-output system as a reference. 

{{< figure src="../img/siso-system-diagram.png" align=center caption="**Figure 8**: Single input single output system diagram." >}}

If we have two inputs \\( x_{1} \\) and \\( x_{2} \\), and if they are correlated, we could think of \\( x_{2} \\) as being made up of an uncorrelated part, \\( x_{2_{UC}} \\) and a part that is correlated with \\( x_{1} \\) which we can call \\( x_{2_{C}} \\). 

We can think of this correlated part as being a linear transformation of \\( x_{1} \\), which we can represent by a transfer function \\( L_{12}(f) \\). Note that often these correlation effects will be non-linear so we will use the notation \\( L_{12}(f) \\) to represent the transfer function with the optimum linear effects between the input \\( x_{1} \\) and output \\( x_{2_{C}} \\).

**Figure 9** shows this idea in a diagram.

{{< figure src="../img/miso-system-diagram.png" align=center caption="**Figure 8**: Two-input single-output system diagram with partially correlated inputs." >}}

Now, if we wanted to find the frequency response between \\( x_{1} \\) and \\( y \\), we would think to calculate

$$
    \frac{Y(f)}{X_{1}(f)}
$$

but because \\( x_{2} \\) also effects the output \\( y \\) through \\( H_{2y} \\) *and* because \\( x_{1} \\) is correlated with \\( x_{2} \\) through \\( L_{12} \\), doing this will actually calculate

$$
    \frac{Y(f)}{X_{1}(f)} = H_{1y}(f) + H_{2y}(f) L_{12}(f)
$$

which does *not* just give us \\( H_{1y}(f) \\) as we want. Because \\( x_{1} \\) gets to \\( y \\) through multiple paths (not just through \\( H_{1y} \\)), we have to account for this 'error' to isolate the actual frequency response we care about: \\( H_{1y}(f) \\).


### Conditioned Frequency Responses

So how do we use the tools we developed to correct, or *condition*, these frequency responses to actually give us what we want? Since we're now defining our frequency response in terms of spectral functions, for our two input example we know we will need \\( G_{1y} \\) and \\( G_{11} \\). Let's recall the definition of our spectral density function using \\( G_{1y} \\) (ignoring the limit as \\( T \to \infty \\)). 

$$
    G_{1y}(f) = \frac{2}{T} E\\{ X^{*}_{1}(f) Y(f) \\}
$$

where we just showed that

$$
    Y(f) = H_{1y}(f) X_{1}(f) + H_{2y}(f) L_{12}(f) X_{1}(f)
$$

Substituing this into our spectral density function gives

$$
\begin{aligned}
    G_{1y}(f) &= \frac{2}{T} E\\{ X^{*}_{1}(f) \big[ H\_{1y}(f) X\_{1}(f) + H\_{2y}(f) L\_{12}(f) X\_{1}(f) \big] \\}
    \\\
    &= \frac{2}{T} E\\{ H\_{1y}(f) X^\*\_{1}(f) X\_{1}(f) + H\_{2y}(f) L\_{12}(f) X^\*\_{1}(f) X\_{1}(f) \\}
    \\\
    &= \frac{2}{T} \bigg( H\_{1y}(f) \frac{T}{2} G\_{11}(f) + H\_{2y}(f) L\_{12}(f) \frac{T}{2} G\_{11}(f) \bigg)
    \\\
    &= H\_{1y}(f) G\_{11}(f) + H\_{2y}(f) L\_{12}(f) G\_{11}(f)
\end{aligned}
$$

From this, we can see that

$$
    \frac{G_{1y}(f)}{G_{11}(f)} = H_{1y}(f) + H_{2y}(f) L_{12}(f)
$$

and therefore

$$
    H_{1y}(f) = \frac{G_{1y}(f)}{G_{11}(f)} - H_{2y}(f) L_{12}(f)
$$

as we saw before.

The important equation here is 

$$
    G_{1y}(f) = H\_{1y}(f) G\_{11}(f) + H\_{2y}(f) L\_{12}(f) G\_{11}(f)
$$

Notice that 

$$
    L_{12}(f) G_{11}(f) = G_{12}(f)
$$

so we can write

$$
    G_{1y}(f) = H\_{1y}(f) G\_{11}(f) + H\_{2y}(f) G_{12}(f)
$$

A very similar method could be used for \\( G_{2y}(f) \\).

$$
    G_{2y}(f) = H\_{1y}(f) G\_{21}(f) + H\_{2y}(f) G_{22}(f)
$$

This gives us two equations with only two unknows: \\( H\_{1y}(f) \\) and \\( H\_{2y}(f) \\), the two frequency responses of the system. We can calculate all the spectral functions here.

We can solve these simultaneously in matrix form for each discrete frequency \\( f_{i} \\).

$$
    \begin{bmatrix}
        G_{11}(f_{i}) & G_{12}(f_{i}) \\\
        G_{21}(f_{i}) & G_{22}(f_{i})
    \end{bmatrix}
    \begin{bmatrix}
        H_{1y}(f_{i}) \\\ H_{2y}(f_{i})
    \end{bmatrix}
    =
    \begin{bmatrix}
        G_{1y}(f_{i}) \\\ G_{2y}(f_{i})
    \end{bmatrix}
$$

$$
    \begin{bmatrix}
        H_{1y}(f_{i}) \\\ H_{2y}(f_{i})
    \end{bmatrix}
    =
    \begin{bmatrix}
        G_{11}(f_{i}) & G_{12}(f_{i}) \\\
        G_{21}(f_{i}) & G_{22}(f_{i})
    \end{bmatrix} ^{-1}
    \begin{bmatrix}
        G_{1y}(f_{i}) \\\ G_{2y}(f_{i})
    \end{bmatrix}
$$

This finally gives us our conditioned frequency responses. We can also extend this idea to an arbitrary number of inputs.

## An example, revisited

Let's go back to the example we started this all with and take a closer look at how the system was
defined. 

***TO BE CONTINUED***


## References

[1] [What is Autocorrelation?](https://www.youtube.com/watch?v=hOvE8puBZK4&t=812s)

[2] Bendat, Julius S., and Allan G. Piersol. Random data: analysis and measurement procedures. John Wiley & Sons, 2011.

[3] Tischler, Mark B., and Robert K. Remple. Aircraft and rotorcraft system identification. Reston, VA: American Institute of Aeronautics and Astronautics, 2012.

[4] Otnes, Robert K. "Digital time series analysis." John Wiley & Sons, (1972).