/*
libfive: a CAD kernel for modeling with implicit functions
Copyright (C) 2018  Matt Keeter

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "catch.hpp"

#include "libfive/render/brep/simplex/simplex_pool.hpp"
#include "libfive/render/brep/simplex/solver.hpp"
#include "libfive/render/brep/simplex/simplex_mesher.hpp"
#include "libfive/render/brep/indexes.hpp"
#include "libfive/render/brep/dual.hpp"
#include "util/shapes.hpp"

using namespace Kernel;

TEST_CASE("SimplexPool")
{
    auto c = circle(1);
    auto r = Region<2>({-1, -1}, {1, 1});

    auto t = SimplexTreePool<2>::build(c, r, 0.1, 1e-8, 1);
    REQUIRE(t.get() != nullptr);

    t->assignIndices();
}

TEST_CASE("SimplexTree<2>::assignIndices")
{
    auto c = circle(1);
    auto r = Region<2>({-1, -1}, {1, 1});

    auto t = SimplexTreePool<2>::build(c, r, 1.1, 1e-8, 1);
    REQUIRE(t.get() != nullptr);

    t->assignIndices();

    REQUIRE(t->isBranch());
    std::set<uint64_t> indices;
    for (auto& c : t->children) {
        REQUIRE(c.load() != nullptr);
        REQUIRE(!c.load()->isBranch());
        REQUIRE(c.load()->leaf != nullptr);
        for (auto& i : c.load()->leaf->index)
        {
            indices.insert(i);
        }
    }
    REQUIRE(indices.size() == 25);
    REQUIRE(*indices.begin() == 1);
    REQUIRE(*indices.rbegin() == 25);
}

TEST_CASE("SimplexTree<3>: types")
{
    auto c = sphere(0.5);
    auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

    auto t = SimplexTreePool<3>::build(c, r);
    REQUIRE(t.get() != nullptr);

    REQUIRE(t->isBranch());
    for (auto& c : t->children) {
        REQUIRE(c.load()->type == Interval::AMBIGUOUS);
    }
}

void test_corner_positions(const SimplexTree<3>* ptr, Region<3> r)
{
    std::list<std::pair<const SimplexTree<3>*, Region<3>>> todo;
    todo.push_back({ptr, r});

    unsigned checked_count = 0;
    while (todo.size())
    {
        auto task = todo.front();
        todo.pop_front();

        if (task.first->isBranch())
        {
            auto rs = task.second.subdivide();
            for (unsigned i=0; i < 8; ++i) {
                todo.push_back({task.first->children[i].load(), rs[i]});
            }
        }
        else if (task.first->leaf)
        {
            for (unsigned i=0; i < 8; ++i) {
                Eigen::Vector3d vt = task.first->leaf->vertices.row(
                        CornerIndex(i).neighbor().i);
                Eigen::Vector3d vr = task.second.corner(i);
                CAPTURE(vt);
                CAPTURE(vr);
                REQUIRE((vt - vr).norm() < 1e-6);
                checked_count++;
            }
        }
    }
    REQUIRE(checked_count > 0);
}

TEST_CASE("SimplexTree<3>: Corner positions")
{
    SECTION("Sphere")
    {
        auto c = sphere(0.5);
        auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

        auto t = SimplexTreePool<3>::build(c, r);
        REQUIRE(t.get() != nullptr);
        test_corner_positions(t.get(), r);
    }

    SECTION("Box (low-resolution)")
    {
        auto c = box({-0.4, -0.4, -0.4}, {0.4, 0.4, 0.4});
        auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

        auto t = SimplexTreePool<3>::build(c, r, 0.4, 0, 1);
        test_corner_positions(t.get(), r);
    }
}

TEST_CASE("SimplexTree<3>::assignIndices")
{
    auto c = sphere(0.5);
    auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

    auto t = SimplexTreePool<3>::build(c, r, 1.1, 1e-8, 1);
    REQUIRE(t.get() != nullptr);

    t->assignIndices();

    REQUIRE(t->isBranch());
    std::set<uint64_t> indices;
    for (auto& c : t->children) {
        REQUIRE(c.load() != nullptr);
        REQUIRE(!c.load()->isBranch());
        REQUIRE(c.load()->leaf != nullptr);
        for (auto& i : c.load()->leaf->index)
        {
            indices.insert(i);
        }
    }
    REQUIRE(indices.size() == 125);
    REQUIRE(*indices.begin() == 1);
    REQUIRE(*indices.rbegin() == 125);
}

TEST_CASE("SimplexTree<3>::leafLevel")
{
    auto c = sphere(0.5);
    auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

    auto t = SimplexTreePool<3>::build(c, r, 1.1, 1e-8, 1);
    REQUIRE(t->isBranch());
    for (auto& c : t->children) {
        REQUIRE(c.load()->leafLevel() == 0);
    }
}

TEST_CASE("SimplexMesher (smoke test)")
{
    auto c = sphere(0.5);
    auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

    auto t = SimplexTreePool<3>::build(c, r, 1.1, 1e-8, 1);
    REQUIRE(t->isBranch());
    for (auto& c : t->children) {
        REQUIRE(c.load()->type == Interval::AMBIGUOUS);
    }
    t->assignIndices();

    std::atomic_bool cancel(false);
    auto m = Dual<3>::walk<SimplexMesher>(t, 8,
            cancel, EMPTY_PROGRESS_CALLBACK, c);

    REQUIRE(m->branes.size() > 0);
    REQUIRE(m->verts.size() > 1);
    m->saveSTL("out.stl");
}

void test_edge_pairs(const Mesh& m) {
    // Every edge must be shared by two triangles
    // We build a bitfield here, counting forward and reverse edges
    std::map<std::pair<int, int>, int> edges;
    for (const auto& t : m.branes) {
        for (unsigned i=0; i < 3; ++i) {
            const auto a = t[i];
            const auto b = t[(i + 1) % 3];
            auto key = std::make_pair(std::min(a, b), std::max(a, b));
            if (!edges.count(key)) {
                edges.insert({key, 0});
            }
            if (a < b)
            {
                REQUIRE((edges[key] & 1) == 0);
                edges[key] |= 1;
            }
            else
            {
                REQUIRE((edges[key] & 2) == 0);
                edges[key] |= 2;
            }
        }
    }
    for (auto& p : edges) {
        CAPTURE(p.first.first);
        CAPTURE(p.first.second);
        CAPTURE(m.verts[p.first.first]);
        CAPTURE(m.verts[p.first.second]);
        REQUIRE(p.second == 3);
    }
}

TEST_CASE("SimplexMesher: edge pairing")
{
    std::atomic_bool cancel(false);
    const unsigned workers = 8;

    SECTION("Sphere (low resolution)")
    {
        auto c = sphere(0.4);
        auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

        auto t = SimplexTreePool<3>::build(c, r, 1.1, 0, 1);
        REQUIRE(t->isBranch());
        for (auto& c : t->children) {
            REQUIRE(c.load()->type == Interval::AMBIGUOUS);
        }
        t->assignIndices();

        auto m = Dual<3>::walk<SimplexMesher>(t, workers,
                cancel, EMPTY_PROGRESS_CALLBACK, c);

        test_edge_pairs(*m);
    }

    SECTION("Sphere (higher-resolution)")
    {
        auto c = sphere(0.4);
        auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

        auto t = SimplexTreePool<3>::build(c, r, 0.4, 0, 1);
        t->assignIndices();

        auto m = Dual<3>::walk<SimplexMesher>(t, workers,
                cancel, EMPTY_PROGRESS_CALLBACK, c);

        test_edge_pairs(*m);
    }

    SECTION("Box (low-resolution)")
    {
        auto c = box({-0.4, -0.4, -0.4}, {0.4, 0.4, 0.4});
        auto r = Region<3>({-1, -1, -1}, {1, 1, 1});

        auto t = SimplexTreePool<3>::build(c, r, 0.4, 0, 1);
        t->assignIndices();

        auto m = Dual<3>::walk<SimplexMesher>(t, workers,
                cancel, EMPTY_PROGRESS_CALLBACK, c);

        test_edge_pairs(*m);
    }
}

TEST_CASE("SimplexMesher: menger sponge")
{
    Tree sponge = max(menger(2), -sphere(1, {1.5, 1.5, 1.5}));
    Region<3> r({-2.5, -2.5, -2.5}, {2.5, 2.5, 2.5});

    auto t = SimplexTreePool<3>::build(sponge, r, 0.1);
    t->assignIndices();

    std::atomic_bool cancel(false);
    auto m = Dual<3>::walk<SimplexMesher>(t,
            8, cancel, EMPTY_PROGRESS_CALLBACK, sponge);

    m->saveSTL("sponge.stl");
}

TEST_CASE("Simplex meshing (gyroid performance breakdown)", "[!benchmark]")
{
    Region<3> r({ -5, -5, -5 }, { 5, 5, 5 });

    Root<SimplexTree<3>> t;
    unsigned workers = 8;
    std::atomic_bool cancel(false);

    auto s = sphereGyroid();

    BENCHMARK("SimplexTree construction")
    {
        t = SimplexTreePool<3>::build(s, r, 0.05, 1e-8, workers);
    }

    BENCHMARK("Assigning indices")
    {
        t->assignIndices();
    }

    std::unique_ptr<Mesh> m;
    BENCHMARK("Mesh building")
    {
        m = Dual<3>::walk<SimplexMesher>(
                t, workers, cancel, EMPTY_PROGRESS_CALLBACK, s);
    }

    BENCHMARK("SimplexTree deletion")
    {
        t.reset();
    }
    m->saveSTL("bench.stl");

    BENCHMARK("Mesh deletion")
    {
        m.reset();
    }
}
