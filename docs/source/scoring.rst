Scoring systems
===============

Extensivity
-----------

Many intuitive scoring criteria for segments, such as fold-enrichment, produce quantities that are "intensive", that is, they do not scale with a segment's length. Since energies (and scores) are extensive quantities, applying an intensive measure produces an implicit *a priori* bias against large segments, because a segmentation with *larger* segments will have *fewer* segments, and therefore a lower cumulative score when fold-enrichment among all segments is identical. 

Therefore, it is important to note that to turn an intensive segment quality metric into a scoring function with a uniform prior on segment length, one must scale a segment's score in direct proportion to its length in bins: a "fair" scoring function must be extensive.

On the other hand, we can impose an explicit prior on segment length by building a scoring function from an intensive quality metric and applying to it a particular scaling relation with respect to segment length.


Log-odds scores
---------------


Potts energy model
------------------

In the space of segmentations :math:`S`, we can decompose the multiresolution :ref:`Potts <equation-potts>` Hamiltonian described  earlier into an energy function for segments:

.. comment
	.. math::
	   E_{potts}(a,b) &= -\frac{1}{m} \sum_{i=a}^b \sum_{j=i}^b \left(A_{ij}  - \gamma\frac{k_i k_j}{2m} \right)

	where :math:`E_{seg}(a,b)` is proportional to the sum of interaction
	scores between nodes in the segment :math:`[a,b)`, and
	:math:`E_{null}(a,b)` is the corresponding expected value under the null
	model.

When a heatmap is a stochastic matrix (e.g. a *balanced* Hi-C heatmap), we can take all :math:`k_i = 1` and :math:`2m = N`. Then the Potts energy function can be written:

.. math::
   S_{\textrm{potts}}(a,b) &= \sum_{i=a}^{b-1}\sum_{j=a}^{b-1} \left(A_{ij}  - \frac{\gamma}{N} \right) \\
   				  &= \left[\sum_{i=a}^{b-1}\sum_{j=a}^{b-1} A_{ij}\right]  - \left[\frac{\gamma}{N}(b-a)^2\right]

Consider the submatrix corresponding to segment :math:`[a,b)` in the heatmap: ``A[a:b,a:b]``. We see that the configuration null model makes an assumption about how much edge mass is dedicated to every such submatrix *per pixel*. The Potts model score takes the difference between the observed mass in the submatrix and this background mass. 

The Potts scoring function imposes a segment length bias on segmentations: Take a segment with total edge mass :math:`c` and a scale it up to twice its size, so that it has twice its original length and total edge mass :math:`c^2`. We can see that the Potts score will increase by a factor of four, rather than two.

.. comment
	For a uniform balanced (i.e. flat) heatmap, the segment score would decrease quadratically with length for :math:`\gamma > 1/N`. The resolution parameter :math:`\gamma` determines the strength of this trend. The exact relationship between 



Other scoring functions
-----------------------

Armatus
~~~~~~~
Filipova et al (2014) presented the optimal segmentation algorithm in the context
of a Hi-C domain scoring function with a tunable scale parameter to find domains
at multiple resolutions.

.. math::
	& q(a,b) = \frac{\sum_{i=a}^{b-1} \sum_{j=a}^{b-1} A_{ij}}{ (b-a)^\gamma }\\
	& \mu(l) = \textrm{mean } q \textrm{ for segments with length } l \\
	\\
	& S_{ \textrm{armatus} }(a,b) = \max \left(0, q(a,b) - \mu(b-a) \right)

Corner score
~~~~~~~~~~~~

Phase-consistency score
~~~~~~~~~~~~~~~~~~~~~~~