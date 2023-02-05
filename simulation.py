import cvxpy as cp
import numpy as np

np.random.seed(0)

m = 10
n = 6

# parameters
c = np.random.randint(20, size=(n, m+1))

alpha = np.random.randint(1, 20, size=(m+1, 1))

R = np.random.randint(20, size=(n-1, m+1))

m_in = np.random.randint(20, size=(n, 1))
m_devices = np.random.randint(20, size=(m+1, 1))

p = np.zeros(((2*n-1)*(m+1), 1))
p[:(m+1)*n, :] = (c / alpha.reshape(1, -1)).T.reshape(-1, 1)

qz = R.T.reshape(-1, 1)

S = -np.eye( (m+1)*(n-1), (m+1)*(2*n-1) ) + np.eye( (m+1)*(n-1), (m+1)*(2*n-1), k=1 ) - np.eye( (m+1)*(n-1), (m+1)*(2*n-1), k=(m+1)*n )

K = 3
h = K * alpha

L = np.zeros( (m+1, (m+1)*(2*n-1)) )
for i in range(m):
  L[i, i*n:(i+1)*n] = c[:, i]

I = np.zeros((m+1)*(n-1))
y = np.ones( ((m+1)*(2*n-1), 1) )
y[-(m+1)*(n-1):, 0] = 0
M = np.zeros( (m+1, (m+1)*(2*n-1)) )
for i in range(m):
  M[i, i*n:(i+1)*n] = m_in.reshape(1, -1)

# optimization variable
z = cp.Variable( ( (2*n-1)*(m+1), 1), boolean=True)
zz = cp.Variable( (m+1)*(n-1), boolean=True)

# print(np.linalg.eigvals(Q))
# print(np.all(np.linalg.eigvals(Q) > 0))

# print(M.shape)
# print(z.shape)
# print(m_devices.shape)

objective = cp.Minimize(p.T @ z + 0.5 * qz.T @ zz)
constrinats = [S @ z <= 0,
               L @ z <= h,
               y.T @ z == I,
               M @ z <= m_devices]

prob = cp.Problem(objective, constrinats)
prob.solve(solver='MOSEK')

print("Status:", prob.status)
print("Optimal value", prob.value)
# print("Optimal var", z.value)
