import cvxpy as cp
import numpy as np

m = 2
n = 3
K = 2

## parameters
C = np.array([[100, 200, 200],
              [100, 200, 200],
              [100, 200, 200]]).reshape((n, m+1))
R = np.array([[10, 20, 30],
              [10, 20, 30]]).reshape((n-1, m+1))
m_l = np.array([100, 100, 300]).reshape((n, 1))
m_d = np.array([500, 500, 500]).reshape((m+1, 1))
alpha = np.array([50, 60, 100]).reshape((m+1, 1))

## middle variables
p = np.zeros(( (m+1)*(2*n-1), 1) )
p[:(m+1)*n, :] = (C / alpha.reshape(1, -1)).T.reshape(-1, 1)

q = 0.5 * R.T.reshape(-1, 1)

S_1 = np.zeros( ((m+1)*(n-1), (m+1)*(2*n-1)) )
for idx in range(m+1):
  temp_idx_x = np.array(range(idx*(n-1), (idx+1)*(n-1)))
  temp_idx_y = np.array(range(idx+idx*(n-1), idx+(idx+1)*(n-1)))
  S_1[temp_idx_x, temp_idx_y] = -np.ones(n-1)
  
  temp_idx_x = np.array(range(idx*(n-1), (idx+1)*(n-1)))
  temp_idx_y = np.array(range(1+idx+idx*(n-1), 1+idx+(idx+1)*(n-1)))
  S_1[temp_idx_x, temp_idx_y] = np.ones(n-1)

  temp_idx_x = np.array(range(idx*(n-1), (idx+1)*(n-1)))
  temp_idx_y = np.array(range((m+1)*(n-1)+1+idx+idx*(n-1), (m+1)*(n-1)+1+idx+(idx+1)*(n-1)))
  S_1[temp_idx_x, temp_idx_y] = -np.ones(n-1)

S_2 = np.zeros( ((m+1)*(n-1), (m+1)*(2*n-1)) )
for idx in range(m+1):
  temp_idx_x = np.array(range(idx*(n-1), (idx+1)*(n-1)))
  temp_idx_y = np.array(range(idx+idx*(n-1), idx+(idx+1)*(n-1)))
  S_2[temp_idx_x, temp_idx_y] = -np.ones(n-1)
  
  temp_idx_x = np.array(range(idx*(n-1), (idx+1)*(n-1)))
  temp_idx_y = np.array(range(1+idx+idx*(n-1), 1+idx+(idx+1)*(n-1)))
  S_2[temp_idx_x, temp_idx_y] = np.ones(n-1)

  temp_idx_x = np.array(range(idx*(n-1), (idx+1)*(n-1)))
  temp_idx_y = np.array(range((m+1)*(n-1)+1+idx+idx*(n-1), (m+1)*(n-1)+1+idx+(idx+1)*(n-1)))
  S_2[temp_idx_x, temp_idx_y] = np.ones(n-1)

G = np.zeros( ((m+1)*(n-1), (m+1)*(2*n-1)) )
for idx in range(m+1):
  temp_idx_x = np.array(range(idx*(n-1), (idx+1)*(n-1)))
  temp_idx_y = np.array(range(1+idx+idx*(n-1), 1+idx+(idx+1)*(n-1)))
  G[temp_idx_x, temp_idx_y] = np.ones(n-1)

  temp_idx_x = np.array(range(idx*(n-1), (idx+1)*(n-1)))
  temp_idx_y = np.array(range((m+1)*(n-1)+1+idx+idx*(n-1), (m+1)*(n-1)+1+idx+(idx+1)*(n-1)))
  G[temp_idx_x, temp_idx_y] = np.ones(n-1)

I_2 = np.ones(((m+1)*(n-1), 1))
h = K * alpha

L = np.zeros( (m+1, (m+1)*(2*n-1)) )
for i in range(m):
  L[i, i*n:(i+1)*n] = C[:, i]

I = np.ones((n, 1))

E = np.zeros( (n, (m+1)*(2*n-1)) )
for i in range(n):
  idx = list(range(i, (m+1)*(n)+i, n))
  E[i, idx] = 1

M = np.zeros( (m+1, (m+1)*(2*n-1)) )
for i in range(m+1):
  M[i, i*n:(i+1)*n] = m_l.reshape(1, -1)
# print(M)
# print(m_d)
V = np.zeros( ((m+1)*(n-1), (m+1)*(2*n-1)) )
V[:(m+1)*(n-1), 1:(m+1)*(n-1)+1] = np.diag( -np.ones((m+1)*(n-1)) )

U = np.zeros( ((m+1)*(n-1), (m+1)*(2*n-1)) )
U[-(m+1)*(n-1):, -(m+1)*(n-1):] = np.diag( -np.ones((m+1)*(n-1)) )

# optimization variable
z = cp.Variable( ( (2*n-1)*(m+1), 1), boolean=True)
w = cp.Variable( ((m+1)*(n-1), 1), boolean=True)

objective = cp.Minimize(p.T @ z + q.T @ w)
constraints = [S_1 @ z <= 0,
               -S_2 @ z <= 0,
               L @ z <= h,
               E @ z == I,
               M @ z <= m_d,
               U @ z + w <= 0,
               V @ z + w <= 0,
               G @ z - w <= I_2]

prob = cp.Problem(objective, constraints)
prob.solve(solver='MOSEK')

print("Status:", prob.status)
print("Optimal value", prob.value)

x = np.array(z.value)[:(m+1)*n].reshape((n, m+1))
print(f"Optimal variable:\n{x}")
