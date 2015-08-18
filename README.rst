Lavaburst
=========

Chromatin domains bursting with flavor!

Let's get started! See `IPython
Notebook <http://nbviewer.ipython.org/github/nezar-compbio/lavaburst/blob/master/example/example.ipynb>`__.

Optimal domain segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Produce the highest scoring domain segmentation according to the model.

Marginal domain boundary probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Produce the marginal probabilities for each bin edge being a domain boundary. The
marginal boundary probability of a single bin edge is the total
probability over all possible domain segmentations for which that bin edge serves as a
border between domains. The output is a 1D array.

Marginal domain probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Produce the marginal probabilities of every unique segment. The output
is a 2D array where ``prob[a,b]`` describes the overall frequency of the
domain spanning bin edges ``a`` and ``b`` to occur in the ensemble of all
possible domain segmentations.

Domain boundary co-occurence probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Produce the marginal co-occurrence probabilities of every pair of bin
edges as domain boundaries. The output is a 2D array where ``prob[a,b]``
describes the overall frequency all segmentations in which both bin
edges ``a`` and ``b`` occur as domain borders.

Locus co-segmentation probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Produce the marginal probability of every pair of bins to co-occur
within the same domain. The output is a 2D array where ``prob[i,j]``
describes the overall probability of genomic bins ``i`` and ``j`` being
part of the same domain, over all possible segmentations.

Exact statistical sampling of domain segmentations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

