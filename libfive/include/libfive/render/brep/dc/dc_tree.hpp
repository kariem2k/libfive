/*
libfive: a CAD kernel for modeling with implicit functions
Copyright (C) 2017  Matt Keeter

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#include <array>
#include <atomic>
#include <iostream>
#include <stack>

#include <cstdint>

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include "libfive/export.hpp"
#include "libfive/eval/eval_xtree.hpp"
#include "libfive/eval/interval.hpp"

#include "libfive/render/brep/region.hpp"
#include "libfive/render/brep/progress.hpp"
#include "libfive/render/brep/xtree.hpp"

#include "libfive/render/brep/dc/intersection.hpp"
#include "libfive/render/brep/dc/marching.hpp"
#include "libfive/render/brep/dc/dc_neighbors.hpp"

namespace Kernel {

/*  AMBIGUOUS leaf cells have more data, which we heap-allocate in
 *  this struct to keep the overall tree smaller. */
template <unsigned N>
struct DCLeaf
{
    DCLeaf();
    void reset();

    /*  level = max(map(level, children)) + 1  */
    unsigned level;

    /*  Vertex locations, if this is a leaf
     *
     *  To make cells manifold, we may store multiple vertices in a single
     *  leaf; see writeup in marching.cpp for details  */
    Eigen::Matrix<double, N, ipow(2, N - 1)> verts;

    /* This array allows us to store position, normal, and value where
     * the mesh crosses a cell edge.  IntersectionVec is small_vec that
     * has enough space for a few intersections, and will move to the
     * heap for pathological cases. */
    std::array<std::shared_ptr<IntersectionVec<N>>, _edges(N) * 2>
        intersections;

    /*  Feature rank for the cell's vertex, where                    *
     *      1 is face, 2 is edge, 3 is corner                        *
     *                                                               *
     *  This value is populated in evalLeaf and used when merging    *
     *  from lower-ranked children                                   */
    unsigned rank;

    /* Used as a unique per-vertex index when unpacking into a b-rep;   *
     * this is cheaper than storing a map of DCTree* -> uint32_t         */
    mutable std::array<uint32_t, ipow(2, N - 1)> index;

    /*  Bitfield marking which corners are set */
    uint8_t corner_mask;

    /*  Stores the number of patches / vertices in this cell
     *  (which could be more than one to keep the surface manifold */
    unsigned vertex_count;

    /*  Marks whether this cell is manifold or not  */
    bool manifold;

    /*  Mass point is the average intersection location *
     *  (the last coordinate is number of points summed) */
    Eigen::Matrix<double, N + 1, 1> mass_point;

    /*  QEF matrices */
    Eigen::Matrix<double, N, N> AtA;
    Eigen::Matrix<double, N, 1> AtB;
    double BtB;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <unsigned N>
class DCTree : public XTree<N, DCTree<N>, DCLeaf<N>>
{
public:
    using Leaf = DCLeaf<N>;

    /*
     *  This is a handle for both the DCTree and the object pool data
     *  that were used to allocate all of its memory.
     */

    /*
     *  Simple constructor
     *
     *  Pointers are initialized to nullptr, but other members
     *  are invalid until reset() is called.
     */
    explicit DCTree();
    explicit DCTree(DCTree<N>* parent, unsigned index);

    /*
     *  Populates type, setting corners, manifold, and done if this region is
     *  fully inside or outside the mode.
     *
     *  Returns a shorter version of the tape that ignores unambiguous clauses.
     */
    std::shared_ptr<Tape> evalInterval(
            IntervalEvaluator& eval, const Region<N>& region,
            std::shared_ptr<Tape> tape);

    /*
     *  Evaluates and stores a result at every corner of the cell.
     *  Sets type to FILLED / EMPTY / AMBIGUOUS based on the corner values.
     *  Then, solves for vertex position, populating AtA / AtB / BtB.
     */
    void evalLeaf(XTreeEvaluator* eval, const DCNeighbors<N>& neighbors,
                  const Region<N>& region, std::shared_ptr<Tape> tape,
                  ObjectPool<Leaf>& spare_leafs);

    /*
     *  If all children are present, then collapse based on the error
     *  metrics from the combined QEF (or interval filled / empty state).
     *
     *  Returns false if any children are yet to come, true otherwise.
     */
    bool collectChildren(
            XTreeEvaluator* eval, std::shared_ptr<Tape> tape,
            double max_err, const Region<N>& region,
            ObjectPool<DCTree<N>>& spare_trees, ObjectPool<Leaf>& spare_leafs);

    /*
     *  Returns the filled / empty state for the ith corner
     */
    Interval::State cornerState(uint8_t i) const;

    /*
     *  Checks whether this cell is manifold.
     *  This must only be called on non-branching cells.
     */
    bool isManifold() const;

    /*
     *  Looks up this cell's corner mask (used in various tables)
     *  This must only be called on non-branching cells.
     */
    uint8_t cornerMask() const;

    /*  Looks up the cell's level.
     *
     *  This must only be called on non-branching cells.
     *
     *  level is defined as 0 for EMPTY or FILLED terminal cells;
     *  for ambiguous leaf cells, it is the number of leafs that
     *  were merged into this cell.
     */
    unsigned level() const;

    /*
     *  Looks up this cell's feature rank.
     *
     *  This must only be called on non-branching cells.
     *
     *  rank is defined as 0 for EMPTY and FILLED cells;
     *  otherwise, it is 1 for a plane, 2 for an edge,
     *  3 for a vertex (in the 3D case).
     */
    unsigned rank() const;

    /*  Boilerplate for an object that contains an Eigen struct  */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /*  Helper typedef for N-dimensional column vector */
    typedef Eigen::Matrix<double, N, 1> Vec;

    /*
     *  Look up a particular vertex by index
     */
    Vec vert(unsigned i=0) const;

    /*
     *  Looks up a particular intersection array by corner indices
     */
    std::shared_ptr<IntersectionVec<N>> intersection(
            unsigned a, unsigned b) const;

protected:
    /*
     *  Searches for a vertex within the DCTree cell, using the QEF matrices
     *  that are pre-populated in AtA, AtB, etc.
     *
     *  Minimizes the QEF towards mass_point
     *
     *  Stores the vertex in vert and returns the QEF error
     */
    double findVertex(unsigned i=0);

    /*
     *  Returns edges (as indices into corners)
     *  (must be specialized for a specific dimensionality)
     */
    const std::vector<std::pair<uint8_t, uint8_t>>& edges() const;


    /*
     *  Writes the given intersection into the intersections list
     *  for the specified edge.  Allocates an interesections list
     *  if none already exists.  The given set of derivatives is normalized
     *  (to become a surface normal).  If the normal is invalid, then
     *  we store an intersection with an all-zero normal.  This means we
     *  can still use the intersection for mass-point calculation, but
     *  can detect that the normal is invalid (and so will not use it for
     *  building the A and b matrices).
     */
    void saveIntersection(const Vec& pos, const Vec& derivs,
                          const double value, const size_t edge);

    /*
     *  Returns a table such that looking up a particular corner
     *  configuration returns whether that configuration is safe to
     *  collapse.
     *  (must be specialized for a specific dimensionality)
     *
     *  This implements the test from [Gerstner et al, 2000], as
     *  described in [Ju et al, 2002].
     */
    static bool cornersAreManifold(const uint8_t corner_mask);

    /*
     *  Checks to make sure that the fine contour is topologically equivalent
     *  to the coarser contour by comparing signs in edges and faces
     *  (must be specialized for a specific dimensionality)
     *
     *  Returns true if the cell can be collapsed without changing topology
     *  (with respect to the leaves)
     */
    static bool leafsAreManifold(
            const std::array<DCTree<N>*, 1 << N>& children,
            const std::array<Interval::State, 1 << N>& corners);

    /*
     *  Returns a corner mask bitfield from the given array
     */
    static uint8_t buildCornerMask(
            const std::array<Interval::State, 1 << N>& corners);

    /*  Eigenvalue threshold for determining feature rank  */
    constexpr static double EIGENVALUE_CUTOFF=0.1f;
};

// Explicit template instantiation declarations
template <> bool DCTree<2>::cornersAreManifold(const uint8_t corner_mask);
template <> bool DCTree<3>::cornersAreManifold(const uint8_t corner_mask);

template <> bool DCTree<2>::leafsAreManifold(
            const std::array<DCTree<2>*, 1 << 2>& children,
            const std::array<Interval::State, 1 << 2>& corners);
template <> bool DCTree<3>::leafsAreManifold(
            const std::array<DCTree<3>*, 1 << 3>& children,
            const std::array<Interval::State, 1 << 3>& corners);

template <> const std::vector<std::pair<uint8_t, uint8_t>>& DCTree<2>::edges() const;
template <> const std::vector<std::pair<uint8_t, uint8_t>>& DCTree<3>::edges() const;

extern template class DCTree<2>;
extern template class DCTree<3>;

}   // namespace Kernel
