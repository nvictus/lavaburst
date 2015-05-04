Examples
========


The segmentation algorithm was run on high quality intra-chromosomal
Hi-C heatmaps with 40kb bins for cell types IMR90 (lung fibroblast) and
H1 (human embryonic stem cell), generated from the raw data of and and
processed using the method of Imakaev . The figure below shows that by
increasing the resolution parameter :math:`\gamma`, one obtains a
monotonic increase in the number of segments (equivalently, the number
of boundary nodes) in the optimal segmentation. This results in a
characteristic \`\`boundary saturation'' curve for each heatmap (see
Figure ).

Note that our formulation of the segmentation problem did not explicity
introduce "gap" segments that occur between high scoring "domain"
segments. Instead, regions with low scores accumulate sequences of
consecutive boundary nodes that imply a lack of discernable domain
structure, as far as the resolution of the data allows.

The y-axis gives the number fraction of nodes :math:`\theta` that are
the starting boundaries of a segment. Because in the Potts model, the
expected number of communities found scales as :math:`\sqrt{\gamma m}` ,
the values on the x-axis are rescaled to make the saturation curves from
different chromsome heatmaps comparable. Interestingly, the curves for
most chromosomes collapse near one another, with characteristic profiles
for the two cell types.

.. raw:: html

   <center>
       

optimal segmentation (blue triangles) at resolution :math:`\gamma=4`,
IMR90 chr22, pixels are log-transformed Hi-C data

.. raw:: html

   </center>

   <center>
       

boundary saturation curves as function of resolution parameter
:math:`\gamma` (optimal segmentations, i.e. zero temperature)

.. raw:: html

   </center>

   <center>
       

marginal boundary distributions at different temperatures
(:math:`1/\beta`)

.. raw:: html

   </center>


   <center>
       

upper triangle: marginal boundary co-occurrence distribution at finite
temperature

.. raw:: html

   </center>

   <center>
       

upper triangle: (log) marginal segment distribution :math:`\gamma = 8`
at finite temperature

.. raw:: html

   </center>

   <center>
       

upper triangle: marginal co-segmentation distribution :math:`\gamma = 6`
at finite temperature

.. raw:: html

   </center>

