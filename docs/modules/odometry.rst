gradslam.odometry
=================================

.. currentmodule:: gradslam.odometry


gradslam.odometry.base
-------------------------------
.. autoclass:: gradslam.odometry.base.OdometryProvider
	:members:

gradslam.odometry.gradicp
-------------------------------
.. autoclass:: GradICPOdometryProvider
	:members:


gradslam.odometry.groundtruth
-------------------------------
.. autoclass:: GroundTruthOdometryProvider
	:members:


gradslam.odometry.icp
-------------------------------
.. autoclass:: ICPOdometryProvider
	:members:


gradslam.odometry.icputils
-------------------------------
.. autofunction:: gradslam.odometry.icputils.solve_linear_system
.. autofunction:: gradslam.odometry.icputils.gauss_newton_solve
.. autofunction:: gradslam.odometry.icputils.point_to_plane_ICP
.. autofunction:: gradslam.odometry.icputils.point_to_plane_gradICP
.. autofunction:: gradslam.odometry.icputils.downsample_pointclouds
.. autofunction:: gradslam.odometry.icputils.downsample_rgbdimages
