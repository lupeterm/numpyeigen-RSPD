#include <npe.h>
#include "stdlib.h"
#include "stddef.h"
#include <iostream>
#include <fstream>
#include <pointcloudio.hpp>
#include <connectivitygraph.h>
#include <planedetector.h>
#include <normalestimator.h>
#include <boundaryvolumehierarchy.h>
#include <point.h>

npe_function(planedetection)
npe_arg(points, dense_float)
npe_arg(normals, dense_float)
npe_begin_code()
{
    std::cout << "(" << points.rows() << ", " << points.cols() << ")" << std::endl;
    std::vector<Point3d> pts;
    for (size_t i = 0; i < points.rows(); i++)
    {
        pts.push_back(Point3d(points.row(i)));
    }
    PointCloud3d pointCloud(pts);
    std::cout << "number of points : " << pointCloud.size() << std::endl;

    // you can skip the normal estimation if you point cloud already have normals
    std::cout << "Estimating normals..." << std::endl;
    std::cout << "hi" << std::endl;
    size_t normalsNeighborSize = 30;

    Octree octree(&pointCloud);

    octree.partition(10, 30);

    ConnectivityGraph *connectivity = new ConnectivityGraph(pointCloud.size());
    pointCloud.connectivity(connectivity);
    NormalEstimator3d estimator(&octree, normalsNeighborSize, NormalEstimator3d::QUICK);
    for (size_t i = 0; i < pointCloud.size(); i++)
    {
        if (i % 10000 == 0)
        {
            std::cout << i / float(pointCloud.size()) * 100 << "%..." << std::endl;
        }
        NormalEstimator3d::Normal normal = estimator.estimate(i);
        connectivity->addNode(i, normal.neighbors);
        pointCloud[i].normal(normal.normal);
        pointCloud[i].normalConfidence(normal.confidence);
        pointCloud[i].curvature(normal.curvature);
    }

    std::cout << "Detecting planes..." << std::endl;
    PlaneDetector detector(&pointCloud);
    detector.minNormalDiff(0.5f);
    detector.maxDist(0.258819f);
    detector.outlierRatio(0.75f);

    std::set<Plane *> planes = detector.detect();
    std::cout << planes.size() << std::endl;

    std::cout << "Saving results..." << std::endl;
    Geometry *geometry = pointCloud.geometry();
    for (Plane *plane : planes)
    {
        geometry->addPlane(plane);
    }
    size_t numPlanes = geometry->numPlanes();
    std::cout <<"numplanes: " <<  numPlanes << std::endl;
    Eigen::MatrixXi planeSizes(numPlanes, 1);
    size_t numPoints = 0;
    for (size_t i = 0; i < numPlanes; i++)
    {
        auto x = geometry->plane(i)->inliers().size();
        planeSizes(i) = x;
        numPoints += x;
    }
    Eigen::MatrixXd allPlanePoints (numPoints, 3);
    std::cout << "(" << allPlanePoints.rows() << ", " << allPlanePoints.cols() << ")" << std::endl;

    std::cout << "numpoints: " << numPoints << std::endl;
    size_t glob_index = 0;
    for (size_t i = 0; i < numPlanes; i++)
    {

        std::cout << "glob_index: " << glob_index << std::endl;
        auto inliers = geometry->plane(i)->inliers();
        for (size_t j = 0; j < planeSizes(i); j++)
        {
            auto d = pointCloud.at(inliers.at(j)).position();
            std::cout << "j:x " << j<< " "<< 0 << std::endl;
            allPlanePoints(glob_index,0) = d(0);  
            std::cout << "j:x " << j<< " "<< 1 << std::endl;
            allPlanePoints(glob_index,1) = d(1);  
            std::cout << "j:x " << j<< " "<< 2 << std::endl;
            allPlanePoints(glob_index,2) = d(2);  
            glob_index += 1;
        }
    }
    std::cout << "planesizes: " << planeSizes << std::endl;
    return std::make_tuple(npe::move(planeSizes), npe::move(allPlanePoints));
}
npe_end_code()
