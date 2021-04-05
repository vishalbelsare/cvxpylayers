import unittest

import cvxpy as cp
import numpy as np
import numpy.random as npr

import jax
import jax.numpy as jnp
from jax import random
from jax.test_util import check_grads
from jax.config import config

from cvxpylayers.jax import CvxpyLayer
import diffcp


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class TestCvxpyLayer(unittest.TestCase):

    def test_example(self):
        key = random.PRNGKey(0)

        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

        key, k1, k2 = random.split(key, num=3)
        A_jax = random.normal(k1, shape=(m, n))
        b_jax = random.normal(k2, shape=(m,))

        # solve the problem
        solution, = cvxpylayer(A_jax, b_jax)

        # compute the gradient of the sum of the solution with respect to A, b
        def sum_sol(A_jax, b_jax):
            solution, = cvxpylayer(A_jax, b_jax)
            return solution.sum()

        dsum_sol = jax.grad(sum_sol)
        dsum_sol(A_jax, b_jax)

    def test_simple_batch_socp(self):
        key = random.PRNGKey(0)
        n = 5
        m = 1
        batch_size = 4

        P_sqrt = cp.Parameter((n, n), name='P_sqrt')
        q = cp.Parameter((n, 1), name='q')
        A = cp.Parameter((m, n), name='A')
        b = cp.Parameter((m, 1), name='b')

        x = cp.Variable((n, 1), name='x')

        objective = 0.5 * cp.sum_squares(P_sqrt @ x) + q.T @ x
        constraints = [A@x == b, cp.norm(x) <= 1]
        prob = cp.Problem(cp.Minimize(objective), constraints)

        prob_jax = CvxpyLayer(prob, [P_sqrt, q, A, b], [x])

        key, k1, k2, k3, k4 = random.split(key, num=5)
        P_sqrt_jax = random.normal(k1, shape=(batch_size, n, n))
        q_jax = random.normal(k2, shape=(batch_size, n, 1))
        A_jax = random.normal(k3, shape=(batch_size, m, n))
        b_jax = random.normal(k4, shape=(batch_size, m, 1))

        def f(*params):
            sol, = prob_jax(*params)
            return sum(sol)

        check_grads(f, (P_sqrt_jax, q_jax, A_jax, b_jax),
                    order=1, modes=['rev'])

    def test_least_squares(self):
        key = random.PRNGKey(0)
        m, n = 100, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_jax = CvxpyLayer(prob, [A, b], [x])

        key, k1, k2 = random.split(key, num=3)
        A_jax = random.normal(k1, shape=(m, n))
        b_jax = random.normal(k2, shape=(m,))

        def lstsq_sum_cp(A_jax, b_jax):
            x = prob_jax(A_jax, b_jax, solver_args={'eps': 1e-10})[0]
            return sum(x)

        def lstsq_sum_linalg(A_jax, b_jax):
            x = jnp.linalg.solve(
                A_jax.T @ A_jax + jnp.eye(n),
                A_jax.T @ b_jax)
            return sum(x)
        # x_lstsq = lstsq(A_jax, b_jax)

        d_lstsq_sum_cp = jax.grad(lstsq_sum_cp, [0,1])
        d_lstsq_sum_linalg = jax.grad(lstsq_sum_linalg, [0,1])

        grad_A_cvxpy, grad_b_cvxpy = d_lstsq_sum_cp(A_jax, b_jax)
        grad_A_lstsq, grad_b_lstsq = d_lstsq_sum_linalg(A_jax, b_jax)

        self.assertAlmostEqual(
            jnp.linalg.norm(grad_A_cvxpy - grad_A_lstsq).item(),
            0.0,
            places=6)
        self.assertAlmostEqual(
            jnp.linalg.norm(grad_b_cvxpy - grad_b_lstsq).item(),
            0.0,
            places=6)



if __name__ == '__main__':
    unittest.main()
