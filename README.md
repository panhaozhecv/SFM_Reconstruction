# SFM-Reconstruction



**A project of Incremental SFM reconstruction based on Opencv and Ceres-Solver** 



### Environment

Ubuntu18.04



### Requirement

opencv4

ceres-solver with its dependency



### Build

```sh
mkdir build
cd build
cmake ..
make -j4
```



### Run

- Edit configure file in ./config/config.toml, the explanation of config.toml is as follows:

​	**project_folder:** Folder of a reconstruction task.There should be a folder names "color"  which stores images in this folder.And      the name of image should be organized as %04.jpg and **starts from 0001.jpg**.

​    **fx,fy,cx,cy:** The intrinsic parameters of camera.I assume that all the cameras have the same intrinsic params.

​    **k1,k2,p1,p2:** The distortion parameters of camera.In this version, these params are not used yet.

​    **mainCamID:** The index of main camera.A main camera means the camera to whose coordinates everything is transformed. 

- Run in ./build folder:

```sh
./run_sfm
```

- I provide with two group of test results in the folder ./test.



### Future Work

- Improve **robustness**.
- **SPEED UP**.Now the speed of feature extracting and exhaustive matching is quite slow.I am considering use gpu as well as multi thread tricks in the future.

