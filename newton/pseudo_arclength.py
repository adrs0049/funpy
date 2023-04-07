#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np
from pylops import LinearOperator

from states.cont_state import ContinuationState
from newton.deflated_residual import DeflatedResidual
from support.Functional import Functional


class PseudoArcContinuationCorrectorPrecond(LinearOperator):
    def __init__(self, Du, theta, dtype=None, *args, **kwargs):
        self.Du = Du
        self.theta = theta

        # Basic setup for the linear operator
        self.shape = tuple([sum(x) for x in zip(self.Du.shape, (1, 1))])
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if self.theta < 1.0:
            return np.hstack((self.Du._matvec(x[:-1]), (1. - self.theta) * x[-1]))
        else:
            return np.hstack((self.Du._matvec(x[:-1]), x[-1]))

    def to_matrix(self):
        # TODO: can we do this faster?
        return self.tosparse()

class PseudoArcContinuationCorrector(LinearOperator):
    """ Implements the linear operator for pseudo-arclength continuation.
    """
    def __init__(self, state, state0, operator, ks=[], dtype=None, *args, **kwargs):
        """ state, state0: ContinuationState objects """
        # TODO somehow get the deflated residual thing into this!
        dorder = kwargs.get('dorder', 0)
        self.Du = DeflatedResidual(state, operator, par=True, ks=ks, dtype=dtype)
        self.Da = self.Du.colloc.diff_a(state)
        self.ds = kwargs.get('ds', 0.05)
        self.state0 = state0

        # Basic setup for the linear operator
        self.shape = tuple([sum(x) for x in zip(self.Du.shape, (1, 1))])
        self.dtype = np.dtype(dtype)
        self.explicit = False

        # weight values
        self.theta = kwargs.get('theta', 0.5)
        # print('θ = %.4g.' % self.theta)
        assert self.theta > 0 and self.theta <= 1, 'Theta %.2f must be inside (0, 1]!' % self.theta

        # last row must be zero if we estimate this from the null-space
        # Don't setup the namespace for this state since it will set itself to a non-zero
        # continuation parameter destroying the null-space computation we want to do here!
        self.direction = kwargs.get('direction', np.zeros_like(state0))

        # Create functional for (u_dot, u)
        #print('\tdirection = ', self.direction.u, ' state = ', state0, ' state = ', state)
        self.functional = Functional(self.direction.u, order=dorder, basis=self.Du.colloc.name)

        # Also deal with the right hand side
        self.b = self.rhs(state)

        # hook up eta - just for debugging
        self.eta = self.Du.eta

    def rhs(self, state):
        # Also deal with the right hand side
        dstate = state - self.state0
        if self.theta < 1.0:
            ret = np.hstack((self.Du.rhs(state),
                             self.theta * self.functional(dstate.u) +
                             (1. - self.theta) * dstate.a * self.direction.a - self.ds))
        else:
            ret = np.hstack((self.Du.rhs(state),
                             self.functional(dstate.u) + dstate.a * self.direction.a - self.ds))
        # print('state = ', state.a, 'func = ', self.theta * self.functional(dstate.u),
        #       ' dstate = ', dstate.a, ' ret = ', ret[-1], ' ds = ', self.ds, ' dst * dir = ', dstate.a * self.direction.a)
        return ret

    def precond(self):
        return PseudoArcContinuationCorrectorPrecond(self.Du.precond(), self.theta)

    def _matvec(self, dx):
        """ Implements y = Ax

            Here the matrix A is given by:

                / D_u F(a, u)      D_a F(a, u)     \
                \ θ dot{u}_0    (1 - θ) \dot{a}_0  /

            where D_u F(a, u) is a n x n matrix:
                  D_a F(a, u) is a n x 1 matrix
                  \dot{u}_0   is a 1 x n matrix
                  \dot{a}_0   is a 1 x 1 matrix

            here \dot{} = d / ds i.e. the derivative with respect to the arclength.
        """
        # The action of this matrix can then be computed by
        if self.theta < 1.0:
            return np.hstack((self.Du._matvec(dx[:-1]) + self.Da * dx[-1],
                              self.theta * self.functional(dx[:-1]) + (1.0 - self.theta) * self.direction.a * dx[-1]))
        else:
            return np.hstack((self.Du._matvec(dx[:-1]) + self.Da * dx[-1],
                              self.functional(dx[:-1]) + self.direction.a * dx[-1]))

    def _rmatvec(self, x):
        """ Implements x = A^H y """
        assert False, ''
        return np.zeros_like(x)

    def to_matrix(self):
        # TODO: can we do this faster?
        return self.tosparse()
