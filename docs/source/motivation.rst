Motivation
==========

What's in a domain?
-------------------

The first large-scale three-dimensional features about chromatin gleaned
from early Hi-C studies were the open and closed regions of chromatin
called compartments (see Figure ). More recent Hi-C mapping identified
high contact density regions within chromatin compartments called
topologically associating domains (TADs) that are largely conserved in
their positions between cell types . The exact boundaries were
determined by training a hidden Markov model against computed tracks of
contact frequency directionality bias. However, as can be seen by eye in
Figure , such regions can be further stratified into regions both larger
and smaller than the ones defined as TADs (green line segments). This
makes it challenging to determine the biological significance of domains
at any particular scale and highlights the importance of assessing
structural signatures at many different scales and resolutions in order
to identify more precisely what features are predictive of biological
function.

.. _figure-1:

.. figure:: static/img/motiv-eig-tads.png
   :figwidth: 75%
   :align: center
   :alt: features in Hi-C maps

   Fig. 1

   Structural features observed in Hi-C data. (a) Compartments (average size :math:`\sim` 5Mb) form the characteristic plaid pattern in the heatmaps (single chromosome shown). The signal is extracted using spectral decomposition. (b) TADs are smaller (median size :math:`\sim` 400-500kb) segments that are strongly conserved between cell types.


Finding communities in a haystack
---------------------------------

Hi-C heatmaps :math:`A_{ij}` provide quantitative readouts for pairs of
genomic coordinate bins :math:`i` and :math:`j`. We can regard a
symmetric heatmap matrix :math:`A` as the adjacency matrix of an
undirected weighted graph, whose nodes are genomic bins (see :ref:`figure-1`). 
The edge weight :math:`A_{ij}` is an interaction
score for bins :math:`i` and :math:`j`. In the case of processed Hi-C
data, this is normally interpreted as the relative contact frequency
between the two genomic loci.

A network-oriented perspective makes it tempting to apply graph
algorithms for node partitioning, also known as *community detection* in
the field of network science, to identify biologically relevant domains.
One of the most popular methods for community detection in networks is
modularity maximization . For a given assignment of nodes to communities
(i.e., partition), an objective function called *modularity* compares
the edge weight between pairs of nodes in the same community to the
expected edge weight between two nodes under a random null model. The
null model most commonly used is called the *configuration null model*,
which consists of the complete ensemble of graphs having a fixed degree
distribution. Under the configuration null model, the expected weight
between two nodes :math:`i` and :math:`j` of degree :math:`k_i` and
:math:`k_j` is :math:`\frac{k_i k_j}{2m}`, where :math:`m` is the total
edge weight of the graph.


.. _figure-2:

.. figure:: static/img/motiv-hic-graph.png
   :figwidth: 75%
   :align: center
   :alt: Hi-C interaction heatmap as a graph

   Fig. 2

   A binned Hi-C contact frequency heatmap can be interpreted as a complete weighted graph (including zero edge weights) between :math:`N` genomic bins.


The multi-resolution form of the modularity function :math:`Q` for a
partition :math:`\pi` of a network's nodes :math:`\{1,2, \ldots, N\}` is

.. _equation-potts:

.. math::  Q(\pi) = \frac{1}{2m}\sum_{\lt i,j \gt} \left(A_{ij} - \gamma \frac{k_i k_j}{2m}\right)\delta(\pi_i,\pi_j). 
   :label: potts

where the summation runs over all pairs of nodes in the same community
(:math:`\delta(\pi_i, \pi_j) = 1` if :math:`\pi_i=\pi_j` and :math:`0`
if not). The parameter :math:`\gamma` acts as a weight on the null
model, and determines the *resolution* of the partition. The larger the
value of :math:`\gamma`, the smaller the typical size of detected
communities.

Models of interacting spin systems from statistical physics have played
a key role in the development of modern algorithms for community
detection. The most well known spin system is the Ising model, which is
a model of ferromagnetism, where magnetic spins are nodes on a lattice
that take on one of two states (up or down) and interact with their
nearest neighbors. Modularity maximization is actually equivalent to
finding the ground state of a particular kind of spin system known as
the Potts model where each node in the network corresponds to a magnetic
spin that can assume one of a number of different spin states (i.e. the
community labels). Edge weights between nodes in the same spin state
(community) are ferromagnetic (stabilizing) coupling energies. The null
model acts as an anti-ferromagnetic (destabilizing) applied field on
interacting spins. Finding the maximum modularity partition of the graph
therefore corresponds to minimizing the Potts Hamiltonian
:math:`H_{potts}(\pi) = -mQ(\pi)` to find the ground state configuration
of the spins.

Restrict the search space
-------------------------

In general, modularity maximization is NP-complete \cite{}, but we gain a
great deal of traction if we restrict the solution space of partitions
of the nodes to **segmentations**, where only chains of consecutive nodes
can be grouped together. This is convenient because many of the
interesting features seen by eye in Hi-C data, such as TADs and their
apparent substrata, are segmental. Not only can we compute a single
point estimate of the community structure from the data, but we can
efficiently extract a great deal of information about entire statistical
ensembles of segmentations by dynamic programming, and thereby
characterize features at multiple scales.