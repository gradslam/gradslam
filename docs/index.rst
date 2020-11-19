gradslam
=================================

gradSLAM is an open-source library aimed at providing implementations of SLAM subsystems for deep learning (DL) practitioners. While several DL practitioners want to use SLAM (or sub-components therein) from the comfort of their favorite DL library (TensorFlow / PyTorch), there are currently no freely available implementations. We aim to fill in that gap with this library.

Another underlying motivation is to release code for the gradSLAM paper, where we demonstrate end-to-end differentiable dense SLAM systems. People should be able to load up their favorite dataset and run differentiable / non-differentiable dense SLAM systems on them.


.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Tutorials

   tutorials

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: API Reference:

   modules/config
   modules/datasets
   modules/geometry
   modules/metrics
   modules/odometry
   modules/slam
   modules/structures
   modules/utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
