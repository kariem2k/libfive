/*
libfive: a CAD kernel for modeling with implicit functions

Copyright (C) 2018  Matt Keeter

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#include "libfive/render/brep/dc/dc_mesher.hpp"
#include "libfive/render/brep/dual.hpp"

namespace Kernel {

template <Axis::Axis A>
void DCMesher::load(const std::array<const XTree<3>*, 4>& ts)
{
    // Exit immediately if we can prove that there will be no
    // face produced by this edge.
    if (std::any_of(ts.begin(), ts.end(),
        [](const XTree<3>* t){ return t->type != Interval::AMBIGUOUS; }))
    {
        return;
    }

    // Sanity-checking that all cells have a Leaf struct allocated
    for (auto& t : ts)
    {
        assert(t->leaf != nullptr);
        (void)t;
    }

    /*  We need to check the values on the shared edge to see whether we need
     *  to add a face.  However, this is tricky when the edge spans multiple
     *  octree levels.
     *
     * In the following diagram, the target edge is marked with an o
     * (travelling out of the screen):
     *      _________________
     *      | 2 |           |
     *      ----o   1, 3    |  ^ R
     *      | 0 |           |  |
     *      ----------------|  --> Q
     *
     *  If we were to look at corners of c or d, we wouldn't be looking at the
     *  correct edge.  Instead, we need to look at corners for the smallest cell
     *  among the function arguments.
     */
    const auto index = std::min_element(ts.begin(), ts.end(),
            [](const XTree<3>* a, const XTree<3>* b)
            { return a->leaf->level < b->leaf->level; }) - ts.begin();

    constexpr auto Q = Axis::Q(A);
    constexpr auto R = Axis::R(A);

    constexpr std::array<uint8_t, 4> corners = {{Q|R, R, Q, 0}};

    // If there is a sign change across the relevant edge, then call the
    // watcher with the segment corners (with proper winding order)
    auto a = ts[index]->cornerState(corners[index]);
    auto b = ts[index]->cornerState(corners[index] | A);
    if (a != b)
    {
        if (a != Interval::FILLED)
        {
            load<A, 0>(ts);
        }
        else
        {
            load<A, 1>(ts);
        }
    }
}

template <Axis::Axis A, bool D>
void DCMesher::load(const std::array<const XTree<3>*, 4>& ts)
{
    int es[4];
    {   // Unpack edge vertex pairs into edge indices
        auto q = Axis::Q(A);
        auto r = Axis::R(A);
        std::vector<std::pair<unsigned, unsigned>> ev = {
            {q|r, q|r|A},
            {r, r|A},
            {q, q|A},
            {0, A}};
        for (unsigned i=0; i < 4; ++i)
        {
            es[i] = XTree<3>::mt->e[D ? ev[i].first  : ev[i].second]
                                   [D ? ev[i].second : ev[i].first];
            assert(es[i] != -1);
        }
    }

    uint32_t vs[4];
    for (unsigned i=0; i < ts.size(); ++i)
    {
        assert(ts[i]->leaf != nullptr);

        // Load either a patch-specific vertex (if this is a lowest-level,
        // potentially non-manifold cell) or the default vertex
        auto vi = ts[i]->leaf->level > 0
            ? 0
            : XTree<3>::mt->p[ts[i]->leaf->corner_mask][es[i]];
        assert(vi != -1);

        // Sanity-checking manifoldness of collapsed cells
        assert(ts[i]->leaf->level == 0 || ts[i]->leaf->vertex_count == 1);

        if (ts[i]->leaf->index[vi] == 0)
        {
            ts[i]->leaf->index[vi] = m.verts.size();

            m.verts.push_back(ts[i]->vert(vi).template cast<float>());
        }
        vs[i] = ts[i]->leaf->index[vi];
    }

    // Handle polarity-based windings
    if (!D)
    {
        std::swap(vs[1], vs[2]);
    }

    // Pick a triangulation that prevents triangles from folding back
    // on each other by checking normals.
    std::array<Eigen::Vector3f, 4> norms;

    // Computes and saves a corner normal.  a,b,c must be right-handed
    // according to the quad winding, which looks like
    //     2---------3
    //     |         |
    //     |         |
    //     0---------1
    auto saveNorm = [&](int a, int b, int c){
        norms[a] = (m.verts[vs[b]] - m.verts[vs[a]]).cross
                   (m.verts[vs[c]] - m.verts[vs[a]]).normalized();
    };
    saveNorm(0, 1, 2);
    saveNorm(1, 3, 0);
    saveNorm(2, 0, 3);
    saveNorm(3, 2, 1);

    // Helper function to push triangles that aren't simply lines
    auto push_triangle = [&](uint32_t a, uint32_t b, uint32_t c) {
        if (a != b && b != c && a != c)
        {
            m.branes.push_back({a, b, c});
        }
    };

    if (norms[0].dot(norms[3]) > norms[1].dot(norms[2]))
    {
        push_triangle(vs[0], vs[1], vs[2]);
        push_triangle(vs[2], vs[1], vs[3]);
    }
    else
    {
        push_triangle(vs[0], vs[1], vs[3]);
        push_triangle(vs[0], vs[3], vs[2]);
    }
}

std::unique_ptr<Mesh> DCMesher::mesh(const Root<XTree<3>>& xtree,
                                 std::atomic_bool& cancel,
                                 ProgressCallback progress_callback)
{
    // Perform marching squares
    auto m = std::unique_ptr<Mesh>(new Mesh());
    DCMesher d(*m);

    if (cancel.load() || xtree.get() == nullptr)
    {
        return nullptr;
    }
    else
    {
        std::atomic_bool done(false);
        auto progress_watcher = ProgressWatcher::build(
                xtree.size(), 1.0f,
                progress_callback, done, cancel);

        Dual<3>::walk(xtree.get(), d, progress_watcher);

        done.store(true);
        delete progress_watcher;

#if DEBUG_OCTREE_CELLS
        // Store octree cells as lines
        std::list<const XTree<3>*> todo = {xtree.get()};
        while (todo.size())
        {
            auto t = todo.front();
            todo.pop_front();
            if (t->isBranch())
                for (auto& c : t->children)
                    todo.push_back(c.get());

            static const std::vector<std::pair<uint8_t, uint8_t>> es =
                {{0, Axis::X}, {0, Axis::Y}, {0, Axis::Z},
                 {Axis::X, Axis::X|Axis::Y}, {Axis::X, Axis::X|Axis::Z},
                 {Axis::Y, Axis::Y|Axis::X}, {Axis::Y, Axis::Y|Axis::Z},
                 {Axis::X|Axis::Y, Axis::X|Axis::Y|Axis::Z},
                 {Axis::Z, Axis::Z|Axis::X}, {Axis::Z, Axis::Z|Axis::Y},
                 {Axis::Z|Axis::X, Axis::Z|Axis::X|Axis::Y},
                 {Axis::Z|Axis::Y, Axis::Z|Axis::Y|Axis::X}};
            for (auto e : es)
                m->line(t->cornerPos(e.first).template cast<float>(),
                        t->cornerPos(e.second).template cast<float>());
        }
#endif
        return m;
    }
}

}   // namespace Kernel
