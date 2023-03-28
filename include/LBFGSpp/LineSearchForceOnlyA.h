/**
 * By wftao (Jiantao Wang).
 * For gradient only cases, i.e. NEB optimization
*/

// Copyright (C) 2023-2024 Jiantao Wang <wftao1995@gmail.com>
// Under MIT license

#pragma once

#include <Eigen/Core>
#include <stdexcept>

namespace LBFGSpp {

///
/// A line search algorithm for graident only
///
///   search steps that change the sign of
///   F'(\lambda) = [\nambda^T f(x) + \lambda u] \cdot u
///   minimize F aka find F' = 0
///
template <typename Scalar>
class LineSearchForceOnlyA
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
    ///
    /// Line search by Nocedal and Wright (2006).
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    Parameters for the L-BFGS algorithm.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
    ///                 Can be ignored for the L-BFGS solver.
    /// \param step     In: The initial step length.
    ///                 Out: The calculated step length.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param grad     In: The current gradient vector.
    ///                 Out: The gradient at the new point.
    /// \param dg       In: The inner product between drt and grad.
    ///                 Out: The inner product between drt and the new gradient.
    /// \param x        Out: The new point moved to.
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, const LBFGSParam<Scalar>& param,
                           const Vector& xp, const Vector& drt, const Scalar& step_max,
                           Scalar& step, Scalar& fx, Vector& grad, Scalar& dg, Vector& x)
    {
        // Check the value of step
        if (step <= Scalar(0))
            throw std::invalid_argument("'step' must be positive");

        if (param.linesearch != LBFGS_LINESEARCH_FORCE_ONLY_A)
            throw std::invalid_argument("'param.linesearch' must be 'LBFGS_LINESEARCH_FORCE_ONLY_A' for LineSearchForceOnlyA");

        // The expansion rate of the step size
        const Scalar expansion = Scalar(2);

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if (dg_init > 0)
            throw std::logic_error("the moving direction increases the objective function value");

        // change max_step according to max_move
        Scalar drmax = drt.cwiseAbs().maxCoeff();
        Scalar cur_max_step = step_max;
        if (cur_max_step * drmax > param.max_step_move) {
            cur_max_step = param.max_step_move / drmax;
        }
        step = step > cur_max_step ? cur_max_step : step;

        int iter = 0;
        // linesearch
        // first we calculate F'(step)
        // step 1: bracketing, find a interval that dg change sign
        Scalar step_lo = 0, step_hi;
        Scalar step_inc = step;

        // if F' <= this value, return, s_curv_coeff can be set to as small as 0.1 or even smaller
        const Scalar test_curv = -param.s_curv_coeff * dg_init;
        const Scalar test_curv_bound = -dg_init; 

        bool brackt = false;
        bool touched = false;
        for (;;) {
            // should raise an error
            if (iter++ >= param.max_linesearch) return;

            x.noalias() = xp + step * drt;
            fx = f(x, grad);

            const Scalar dg = grad.dot(drt);

            if (fabs(dg) < test_curv) goto appdamp;
            if (dg > 0) {
                step_hi = step;
                break;
            }
            if (touched) {
                goto appdamp;
            }

            step += step_inc;
            if (step > cur_max_step) {
                step = cur_max_step;
                touched = true;
            }
        }
        // brackting
        step = (step_lo + step_hi) * 0.5;
        for (;;) {
            // should raise an error
            if (iter++ >= param.max_linesearch) return;
            
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
            const Scalar dg = grad.dot(drt);

            if (fabs(dg) < test_curv) goto appdamp;

            if (dg < 0) {
                step_lo = step;
            } else {
                step_hi = step;
            }
            step = (step_lo + step_hi) * 0.5;
        }
appdamp:
        // multply damping before add to x vector
        if (param.damping > 0) {
            step *= param.damping;
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
        }
    }
};

}  // namespace LBFGSpp
