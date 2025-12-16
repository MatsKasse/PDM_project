import numpy as np
import scipy.sparse as sp
import osqp

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def linearize_unicycle(xbar, ubar, Ts):
    """
    x = [px, py, theta], u = [v, w]
    f(x,u) = [px + Ts*v*cos(theta),
              py + Ts*v*sin(theta),
              theta + Ts*w]
    Linearize around (xbar, ubar): x+ = A x + B u + c
    """
    px, py, th = xbar
    v, w = ubar

    #calculate the Jacobians A and B
    A = np.eye(3)
    A[0,2] = -Ts * v * np.sin(th)
    A[1,2] =  Ts * v * np.cos(th)


    B = np.zeros((3,2))
    B[0,0] = Ts * np.cos(th)
    B[1,0] = Ts * np.sin(th)
    B[2,1] = Ts

    #calculate c --> Corrects the linearalized equation.
    f = np.array([
        px + Ts*v*np.cos(th),
        py + Ts*v*np.sin(th),
        th + Ts*w
    ])
    c = f - A@xbar - B@ubar
    return A, B, c



class LinearMPCOSQP:
    """
    QP-solvers only accept this form:
    Min 0.5 * z^T * H * z + q^T * z
    z is here the 'decision variable' that is a stacked formulation of the states and actions on the horizon:
      z = [x0..xN, u0..u_{N-1}]
    The quadratic terms are going to be in H.
    The linear terms are going to be in q.
    H basically is a matrix including Q P and R in this form:
    [Q        ]
    [  Q      ]
    [    P    ]
    [      R  ]
    [        R]
    The vector q pushes the solution towards the reference.
    
    s.t. dynamics equalities, box constraints on u
    """
    def __init__(self, Ts=0.1, N=20,
                 Q=np.diag([20.0, 20.0, 2.0]),
                 R=np.diag([0.5, 0.2]),
                 P=np.diag([30.0, 30.0, 5.0]),
                 vmin=0.0, vmax=0.4, wmax=1.2):

        self.Ts = float(Ts) # Sample time MPC
        self.N = int(N) # horizon length
        self.nx = 3 #amount of states
        self.nu = 2 #amount of inputs

        self.Q = Q # tracking penalty
        self.R = R # control penalty
        self.P = P # terminal penalty

        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.wmax = float(wmax)

        self.prob = osqp.OSQP() #OSQP solver object
        self._is_setup = False #parameter

        
        self.z_prev = None # warm start storage

    def build_qp(self, x_init, x_ref, u_ref):   #convert mpc formulation into OSQP required QP formulation
        """
        x_ref: (N+1,3)
        u_ref: (N,2)
        """
        N, nx, nu = self.N, self.nx, self.nu

        # --- linearization trajectory (simple: around reference) ---
        A_list, B_list, c_list = [], [], []
        for k in range(N):
            A, B, c = linearize_unicycle(x_ref[k], u_ref[k], self.Ts)
            A_list.append(A); B_list.append(B); c_list.append(c)

        # --- decision variable length ---
        nX = (N+1)*nx
        nU = N*nu
        nZ = nX + nU # Decision variable has all states and actions of the horizon in one vector.

        # --- cost H, q ---
        Hx_blocks = [sp.kron(sp.eye(N), sp.csc_matrix(self.Q)), sp.csc_matrix(self.P)] 
        Hx = sp.block_diag(Hx_blocks, format='csc')  # State Cost
        Hu = sp.kron(sp.eye(N), sp.csc_matrix(self.R), format='csc')    # Control Cost
        H = sp.block_diag([Hx, Hu], format='csc') #Cost matrix

        # linear term: -2*Q*x_ref etc. (in 0.5 z^T H z + q^T z form)
        qx = np.zeros(nX)
        for k in range(N):
            qx[k*nx:(k+1)*nx] = -self.Q @ x_ref[k]
        qx[N*nx:(N+1)*nx] = -self.P @ x_ref[N]
        qu = np.zeros(nU)
        for k in range(N):
            qu[k*nu:(k+1)*nu] = -self.R @ u_ref[k]
        q = np.concatenate([qx, qu])

        # --- constraints ---
        # 1) Dynamics equalities:
        # x0 = x_init
        # x_{k+1} - A_k x_k - B_k u_k = c_k
        nEq = (N+1)*nx  # include x0 equality
        Aeq = sp.lil_matrix((nEq, nZ))

        # x0 = x_init
        Aeq[0:nx, 0:nx] = sp.eye(nx)

        # dynamics
        for k in range(N):
            row = (k+1)*nx
            col_xk   = k*nx
            col_xkp1 = (k+1)*nx
            col_uk   = nX + k*nu

            Aeq[row:row+nx, col_xkp1:col_xkp1+nx] = sp.eye(nx)
            Aeq[row:row+nx, col_xk:col_xk+nx] = -A_list[k]
            Aeq[row:row+nx, col_uk:col_uk+nu] = -B_list[k]

        Aeq = Aeq.tocsc()
        leq = np.zeros(nEq)
        ueq = np.zeros(nEq)

        leq[0:nx] = x_init
        ueq[0:nx] = x_init

        for k in range(N):
            row = (k+1)*nx
            leq[row:row+nx] = c_list[k]
            ueq[row:row+nx] = c_list[k]

        # 2) Input box constraints (vmin<=v<=vmax, |w|<=wmax)
        # Only on u part, implemented via additional rows selecting u
        nIneq = nU
        Aineq = sp.lil_matrix((nIneq, nZ))
        Aineq[:, nX:nX+nU] = sp.eye(nU)
        Aineq = Aineq.tocsc()

        l_in = np.zeros(nU)
        u_in = np.zeros(nU)
        for k in range(N):
            l_in[k*nu + 0] = self.vmin
            u_in[k*nu + 0] = self.vmax
            l_in[k*nu + 1] = -self.wmax
            u_in[k*nu + 1] =  self.wmax

        # stack constraints
        Acons = sp.vstack([Aeq, Aineq], format='csc')
        l = np.concatenate([leq, l_in])
        u = np.concatenate([ueq, u_in])

        return H, q, Acons, l, u, nZ

    def solve(self, x_init, x_ref, u_ref):
        H, q, A, l, u, nZ = self.build_qp(x_init, x_ref, u_ref)

            # 1) eerste keer of sparsity veranderd? -> volledige setup
        if (not self._is_setup) or (getattr(self, "_A_nnz", None) != A.nnz):
            self.prob = osqp.OSQP()
            self.prob.setup(P=H, q=q, A=A, l=l, u=u,
                            warm_start=True, verbose=False, polish=True)
            self._is_setup = True
            self._A_nnz = A.nnz
        else:
            # 2) alleen values updaten (zelfde sparsity)
            # H sparsity is constant bij vaste Q/R/P, maar je mag Px ook weglaten als H constant is.
            self.prob.update(Px=H.data, q=q, Ax=A.data, l=l, u=u)

        # warm start (optioneel maar scheelt tijd)
        if self.z_prev is not None and len(self.z_prev) == nZ:
            self.prob.warm_start(x=self.z_prev)

        res = self.prob.solve()
        if res.info.status_val not in (1, 2):  # 1=solved, 2=solved inaccurate
            # fallback: stop veilig
            return np.array([0.0, 0.0]), res

        z = res.x
        self.z_prev = z

        # extract u0
        nx = self.nx
        nX = (self.N+1)*nx
        u0 = z[nX:nX+self.nu]
        return u0, res
