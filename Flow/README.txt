
#Running flow disparity

1. Check CMake for OpenCV libray  directory

2. Change image directory in sgm_stereo.cpp(be careful of gray image used)

3. Change "DIS_FACTOR" at top of SGM.h for visualization.

4. For some cases, <= 4*min_dist gets stable results. (The matchings smoothly spreading over the image is much better. 
   Too small coefficients reduces total number of matchings. Then regions without matchings would receive very bad quality. Here bigger 4 is suggested.)

5. Key places worthy to check: fundamental matrix, rotation matrix, cost values of certain points, accumulated of certain points.

6. After each aggregations, set accumulation to 0 according to flags, making sure no effect along boundary pixels.

7. Add backward computation and consistency check. 

8. Recomputed fundamental matrix imporved the stability.

9. cd build; cmake ../; make; ./sgm_flow

