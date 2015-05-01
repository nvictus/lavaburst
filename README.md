# Lavaburst #

Chromatin interactions bursting with flavor!

Let's get started!

```python

import numpy as np
import seaborn as sns; sns.set_style('white')
blues = sns.cubehelix_palette(0.4, gamma=0.5, rot=-0.3, dark=0.1, light=0.9, as_cmap=True)

from lavaburst.plotting import tilt_matrix, blocks_outline, reveal_tril, reveal_triu
from lavaburst.models import ModularitySegModel
import lavaburst
```

Load a Hi-C heatmap into a numpy array and create a segmentation model instance.

```python
A = np.loadtxt('<Hi-C-matrix-file>')
mask = A.astype(bool).sum(axis=0) > 100  # filter for bins with sufficient counts

model = ModularitySegModel(A, mask, gamma=4.0)
```

### Optimal segmentation ###

Produce the highest scoring segmentation according to the model. The output is a 
monotonic sequence of bin edges, `path`, corresponding to the borders of the segments.

```python
path, scores = model.optimal_segmentation()
segments = zip(path[:-1], path[1:])

f = plt.figure()
ax = f.add_subplot(111)
ax.matshow(np.log(A), cmap=plt.cm.jet)
ax.plot(*blocks_outline(segments), color='k', lw=1)
ax.set_xlim([0, len(A)])
ax.set_ylim([len(A), 0])
```
![optimal segmentation](./example/optimal.png)


### Marginal boundary probabilities ###

Produce the marginal boundary probabilities of every bin edge. The marginal boundary
probability of a single bin edge is the total probability of all segmentations for
which that bin edge serves as a border between segments. The output is a 1D array.

```python
beta = 10000
prob = model.boundary_marginals(beta)
At = tilt_heatmap(A, n_diags=300)

f = plt.figure(figsize=(15,15))
ax = f.add_subplot(111)
ax.matshow(np.log(At), cmap=blues)
ax.set_aspect(np.sqrt(0.25))
ax.set_ylim([300, -100])
```
![boundary marginal](./example/boundary.png)


### Marginal genomic segment probabilities ###

Produce the marginal probabilities of every unique segment. The output is a 2D array
where `prob[a,b]` describes the overall frequency of the segment spanning bin edges `a` and `b`
to occur in the ensemble all possible segmentations.

```python
# Segment marginal distribution
beta = 10000
prob = model.segment_marginals(beta)

f = plt.figure()
ax = f.add_subplot(111)
ax.matshow(reveal_triu(np.log(A)))
cs = ax.contourf(
	reveal_tril(np.log(Ps)), [-200, -100, -50, -10, -5, 0],
	origin='upper', cmap=plt.cm.jet)
f.colorbar(cs)
```
![segment marginal](./example/segment.png)


### Boundary co-occurence probabilities ###

Produce the marginal co-occurrence probabilities of every pair of bin edges as segment boundaries. 
The output is a 2D array where `prob[a,b]` describes the overall frequency all segmentations in
which both bin edges `a` and `b` occur as segment borders.

```python
# Boundary co-occurrence marginal distribution
beta = 10000
prob = model.boundary_cooccur_marginals(beta)

f = plt.figure()
ax = f.add_subplot(111)
ax.matshow(reveal_tril(np.log(A)))
ax.matshow(reveal_triu(prob))
```
![boundary co-marginal](./example/coboundary.png)


### Locus co-segmentation probabilities ###

Produce the marginal probability of every pair of bins to co-occur within the same segment.
The output is a 2D array where `prob[i,j]` describes the overall probability of genomic bins `i`
and `j` being part of the same segment, over all possible segmentations.

```python
# Co-segmentation marginal distribution
beta = 10000
prob = model.segment_cooccur_marginals(beta)

f = plt.figure()
ax = f.add_subplot(111)
ax.matshow(reveal_tril(np.log(A)))
ax.matshow(reveal_triu(prob))
```
![segment co-marginal](./example/cosegment.png)


### Monte Carlo simulation ###

```python
# We use Monte Carlo simulations to get the distribution
# of the number of boundaries in the ensemble at a given temperature
beta = 10000
states = model.metropolis_mcmc(beta)
counts, bin_edges = np.histogram(np.arange(len(A)), states.sum(axis=1))
f = plt.figure()
ax = f.add_subplot(111)
ax.bar(counts)
```

