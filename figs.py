from pylab import *
from matplotlib.patches import FancyArrow

# Foreman-Mackey's taste in figures
rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)

import burstmodel

t = linspace(-10., 10., 10001)
plot(t, burstmodel.event_rate(t, 1., 0.5, 1., 3.), linewidth=2)
xlabel('Time (seconds)')
ylabel('Poisson Rate')
ylim([0., 1.1])
axvline(1., color='r', linestyle='--')
title('A Word')

# Build an arrow.
ar1 = FancyArrow(1., exp(-1.), -0.5, 0., length_includes_head=True,
		color='k', width=0.01)
ar2 = FancyArrow(1., exp(-1.), 1.5, 0., length_includes_head=True,
		color='k', width=0.01)
ax = gca()
# Add the arrow to the axes.
ax.add_artist(ar1)
ax.add_artist(ar2)

show()

savefig('documents/word.pdf', bbox_inches='tight')


