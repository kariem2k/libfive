#include "catch.hpp"

#include "ao/render/brep/xtree.hpp"
#include "util/shapes.hpp"

using namespace Kernel;

TEST_CASE("XTree<2>::vert")
{
    SECTION("Vertex positioning (with two planes)")
    {
        Tree a = min(Tree::X(), Tree::Y());
        auto ta = XTree<2>::build(a, Region<2>({-3, -3}, {1, 1}));
        REQUIRE(ta->vert.x() == Approx(0.0));
        REQUIRE(ta->vert.y() == Approx(0.0));
    }
}

TEST_CASE("XTree<2>::type")
{
    SECTION("Empty")
    {
        Tree a = min(Tree::X(), Tree::Y());
        auto e = XTree<2>::build(a, Region<2>({1, 1}, {2, 2}));
        REQUIRE(e->type == Interval::EMPTY);
    }

    SECTION("Filled")
    {
        Tree a = min(Tree::X(), Tree::Y());
        auto e = XTree<2>::build(a, Region<2>({-3, -3}, {-1, -1}));
        REQUIRE(e->type == Interval::FILLED);
    }

    SECTION("Containing corner")
    {
        Tree a = min(Tree::X(), Tree::Y());
        auto ta = XTree<2>::build(a, Region<2>({-3, -3}, {1, 1}));
        REQUIRE(ta->type == Interval::AMBIGUOUS);
    }
}

TEST_CASE("XTree<2>::isBranch")
{
    SECTION("Empty")
    {
        Tree a = min(Tree::X(), Tree::Y());
        auto e = XTree<2>::build(a, Region<2>({1, 1}, {2, 2}));
        REQUIRE(!e->isBranch());
    }

    SECTION("Filled")
    {
        Tree a = min(Tree::X(), Tree::Y());
        auto e = XTree<2>::build(a, Region<2>({-3, -3}, {-1, -1}));
        REQUIRE(!e->isBranch());
    }

    SECTION("Containing line")
    {
        auto e = XTree<2>::build(Tree::X(), Region<2>({-2, -2}, {2, 2}));
        REQUIRE(!e->isBranch());
    }

    SECTION("Containing corner")
    {
        Tree a = min(Tree::X(), Tree::Y());
        auto ta = XTree<2>::build(a, Region<2>({-3, -3}, {1, 1}));
        REQUIRE(!ta->isBranch());
    }

    SECTION("Containing shape")
    {
        Evaluator e = circle(0.5);
        auto t = XTree<2>::build(circle(0.5), Region<2>({-1, -1}, {1, 1}));
        REQUIRE(t->isBranch());
    }
}

TEST_CASE("XTree<3>::vert")
{
    auto walk = [](std::unique_ptr<const XTree<3>>& xtree, Evaluator& eval){
        std::list<const XTree<3>*> todo = {xtree.get()};
        while (todo.size())
        {
            auto t = todo.front();
            todo.pop_front();
            if (t->isBranch())
            {
                for (auto& c : t->children)
                {
                    todo.push_back(c.get());
                }
            }
            if (!t->isBranch() && t->type == Interval::AMBIGUOUS)
            {
                CAPTURE(t->vert.transpose());
                REQUIRE(eval.eval(t->vert.template cast<float>()) == Approx(0).epsilon(0.001));
            }
        }
    };

    SECTION("Sliced box")
    {
        auto b = max(box({0, 0, 0}, {1, 1, 1}),
                Tree::X() + Tree::Y() + Tree::Z() - 1.3);
        Evaluator eval(b);
        Region<3> r({-2, -2, -2}, {2, 2, 2});
        auto xtree = XTree<3>::build(b, r, 0.1);
        walk(xtree, eval);
    }

    SECTION("Another sliced box")
    {
        auto b = max(box({0, 0, 0}, {1, 1, 1}),
                Tree::X() + Tree::Y() + Tree::Z() - 1.2);
        Evaluator eval(b);
        Region<3> r({-10, -10, -10}, {10, 10, 10});
        auto xtree = XTree<3>::build(b, r, 0.1);
        walk(xtree, eval);
    }
}
