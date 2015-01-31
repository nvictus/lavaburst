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

.. math::
   E_{potts}(a,b) &= -\frac{1}{m} \sum_{i=a}^b \sum_{j=i}^b \left(A_{ij}  - \gamma\frac{k_i k_j}{2m} \right)  \\
          &= E_{seg}(a,b) - \gamma E_{null}(a,b)

where :math:`E_{seg}(a,b)` is proportional to the sum of interaction
scores between nodes in the segment :math:`[a,b)`, and
:math:`E_{null}(a,b)` is the corresponding expected value under the null
model.

.. raw:: html

   <center>

.. raw:: html

   </center>

We see that the Potts model is equivalent to scoring individual segments
according to a log-likelihood ratio against a background distribution.

In particular, the configuration background model enforces a
characteristic segment size (resolution).


Other scoring functions
-----------------------

Armatus
~~~~~~~
Filipova et al (2014) presented the optimal segmentation algorithm in the context
of a Hi-C domain scoring function with a tunable scale parameter to find domains
at multiple resolutions.

Corner score
~~~~~~~~~~~~

Phase-consistency score
~~~~~~~~~~~~~~~~~~~~~~~