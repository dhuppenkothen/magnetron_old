from pylab import *

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
savefig('documents/word.pdf', bbox_inches='tight')


