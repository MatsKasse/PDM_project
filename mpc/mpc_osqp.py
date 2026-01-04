import numpy as np
import scipy.sparse as sp
import osqp



def linearize_unicycle_sincos(xbar, ubar, Ts):
    """
    x = [px, py, sin(theta), cos(theta)], u = [v, w]
    f(x,u) = [px + Ts*v*cos(theta),
              py + Ts*v*sin(theta),
              sin(theta) + Ts*w*cos(theta),
              cos(theta) - Ts*w*sin(theta)]
    Linearize around (xbar, ubar): x = A x + B u + c
    """
    px, py, s, c = xbar
    v, w = ubar

    A = np.eye(4)
    A[0,3] = Ts * v
    A[1,2] = Ts * v
    A[2,3] = Ts * w
    A[3,2] = -Ts * w

    B = np.zeros((4,2))
    B[0,0] = Ts * c
    B[1,0] = Ts * s
    B[2,1] = Ts * c
    B[3,1] = -Ts * s

    f = np.array([
        px + Ts * v * c,
        py + Ts * v * s,
        s + Ts * w * c,
        c - Ts * w * s,
    ])
    c_vec = f - A @ xbar - B @ ubar
    return A, B, c_vec


class LinearMPCOSQP:
    """
    QP-solvers only accept this form:
    Min 0.5 * z^T * H * z + q^T * z
    constraints     l <= A@z <= u
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
        self.nx = 4 #amount of states: [px, py, sin(theta), cos(theta)]
        self.nu = 2 #amount of inputs

        self.Q = Q # tracking penalty
        self.R = R # control penalty
        self.P = P # terminal penalty

        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.wmax = float(wmax)

        self.use_collision_slack = True
        self.slack_weight = 500.0

        self.prob = osqp.OSQP() #OSQP solver object
        self._is_setup = False #parameter
        
        self.z_prev = None # warm start storage

    def _build_dynamics_constraints(self, x_init, A_list, B_list, c_list, nZ):
        """Build equality constraints for dynamics: x0 = x_init and x_{k+1} = A_k x_k + B_k u_k + c_k"""
        N, nx, nu = self.N, self.nx, self.nu
        nX = (N+1)*nx
        
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
        
        return Aeq, leq, ueq

    def _build_control_constraints(self, nZ):
        """Build inequality constraints for control inputs: vmin<=v<=vmax, |w|<=wmax"""
        N, nx, nu = self.N, self.nx, self.nu
        nX = (N+1)*nx
        nU = N*nu
        
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
        
        return Aineq, l_in, u_in
    

    def _count_collision_constraints(self, obs_pred):
        if obs_pred is None:
            return 0
        return sum(len(step) for step in obs_pred)

    def _build_collision_constraints(self, nZ, xbar_xy, obs_pred, slack_offset=None, eps=1e-6):
        """
        Tangent half-space constraints:
        n^T p >= n^T o + r_safe - s
        """
        N, nx = self.N, self.nx

        nrows = self._count_collision_constraints(obs_pred)
        if nrows == 0:
            return None, None, None

        Aineq = sp.lil_matrix((nrows, nZ))
        l = np.zeros(nrows)
        u = np.full(nrows, np.inf)

        row = 0
        for k in range(N + 1):
            xk_idx = k * nx + 0
            yk_idx = k * nx + 1

            xbar = float(xbar_xy[k, 0])
            ybar = float(xbar_xy[k, 1])

            for (ox, oy, r_safe) in obs_pred[k]:
                dx = xbar - ox
                dy = ybar - oy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < eps:
                    dx, dy, dist = 1.0, 0.0, 1.0

                nx_hat = dx / dist
                ny_hat = dy / dist

                Aineq[row, xk_idx] = nx_hat
                Aineq[row, yk_idx] = ny_hat

                if slack_offset is not None:
                    Aineq[row, slack_offset + row] = -1.0

                l[row] = nx_hat * ox + ny_hat * oy + r_safe
                row += 1

        return Aineq.tocsc(), l, u



    def build_qp(self, x_init, x_ref, u_ref, obs_pred=None, xbar_xy=None):
        """
        Convert MPC formulation into OSQP required QP formulation.
        x_ref: (N+1,3)
        u_ref: (N,2)
        l_list, u_list are lists of lower and upper bounds for each constraint block.
        Aeq, leq, ueq are equality constraint matrix and bounds for the dynamics
        xbar is the state from the previous solution. xbar_xy are the x and y part of the xbar
        Acol, l_col, u_col are collision constraints matrix and bounds
        Acons is the final stacked constraint matrix
        """
        N, nx, nu = self.N, self.nx, self.nu

        # --- linearization trajectory (simple: around reference) ---
        A_list, B_list, c_list = [], [], []
        for k in range(N):
            A, B, c = linearize_unicycle_sincos(x_ref[k], u_ref[k], self.Ts)
            A_list.append(A); B_list.append(B); c_list.append(c)

        # --- decision variable length ---
        nX = (N+1)*nx
        nU = N*nu
        nC = 0
        if self.use_collision_slack and obs_pred is not None:
            nC = self._count_collision_constraints(obs_pred)
        nZ = nX + nU + nC # Decision variable has all states and actions of the horizon in one vector.

        # --- cost H, q ---
        Hx_blocks = [sp.kron(sp.eye(N), sp.csc_matrix(self.Q)), sp.csc_matrix(self.P)] 
        Hx = sp.block_diag(Hx_blocks, format='csc')  # State Cost
        Hu = sp.kron(sp.eye(N), sp.csc_matrix(self.R), format='csc')    # Control Cost
        if nC > 0:
            Hs = sp.eye(nC, format='csc') * (2.0 * self.slack_weight)
            H = sp.block_diag([Hx, Hu, Hs], format='csc') #Cost matrix
        else:
            H = sp.block_diag([Hx, Hu], format='csc') #Cost matrix

        # linear term: -2*Q*x_ref etc. (in 0.5 z^T H z + q^T z form)
        qx = np.zeros(nX)
        for k in range(N):
            qx[k*nx:(k+1)*nx] = -self.Q @ x_ref[k]
        qx[N*nx:(N+1)*nx] = -self.P @ x_ref[N]
        qu = np.zeros(nU)
        for k in range(N):
            qu[k*nu:(k+1)*nu] = -self.R @ u_ref[k]
        if nC > 0:
            q = np.concatenate([qx, qu, np.zeros(nC)])
        else:
            q = np.concatenate([qx, qu])

        # --- Build constraints modularly ---
        constraint_list = []
        l_list = []
        u_list = []
        
        # 1) Dynamics constraints (always included)
        Aeq, leq, ueq = self._build_dynamics_constraints(x_init, A_list, B_list, c_list, nZ)
        constraint_list.append(Aeq)
        l_list.append(leq)
        u_list.append(ueq)
        
        # 2) Control input constraints (always included)
        Aineq, l_in, u_in = self._build_control_constraints(nZ)
        constraint_list.append(Aineq)
        l_list.append(l_in)
        u_list.append(u_in)
      


        # 3) Collision constraint(optional)
        if obs_pred is not None:
            if xbar_xy is None:
                xbar_xy = x_ref[:, 0:2]

            slack_offset = (nX + nU) if nC > 0 else None
            Acol, l_col, u_col = self._build_collision_constraints(
                nZ, xbar_xy, obs_pred, slack_offset=slack_offset
            )
            if Acol is not None:
                constraint_list.append(Acol)
                l_list.append(l_col)
                u_list.append(u_col)

            if nC > 0:
                A_slack = sp.lil_matrix((nC, nZ))
                for i in range(nC):
                    A_slack[i, slack_offset + i] = 1.0
                constraint_list.append(A_slack.tocsc())
                l_list.append(np.zeros(nC))
                u_list.append(np.full(nC, np.inf))



        # Stack all constraints
        Acons = sp.vstack(constraint_list, format='csc')
        l = np.concatenate(l_list) 
        u = np.concatenate(u_list)

        return H, q, Acons, l, u, nZ

    def solve(self, x_init, x_ref, u_ref, obs_pred=None):
        xbar_xy = None
        if self.z_prev is not None:
            nx = self.nx
            nX = (self.N + 1) * nx
            xbar = self.z_prev[:nX].reshape(self.N + 1, nx)
            xbar_xy = xbar[:, 0:2]

        H, q, A, l, u, nZ = self.build_qp(x_init, x_ref, u_ref, obs_pred=obs_pred, xbar_xy=xbar_xy)

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
        if res.info.status_val not in (1, 2):  # 1=solved, 2=solved inaccurate (information)
            # fallback: stop veilig als status = 3 of 4
            return np.array([0.0, 0.0]), res

        z = res.x # optimal solution vector z
        self.z_prev = z



        # extract u0
        nx = self.nx
        nX = (self.N+1)*nx
        u0 = z[nX:nX+self.nu] # extract first controll input [v,w]
        return u0, res


def predict_dynamic_obstacles(obstacles, t_now, N, Ts_mpc, robot_radius=0.3, margin=0.2):
    """
    Returns list length (N+1). Each entry is list of (ox, oy, r_safe).
    """
    obs_pred = [[] for _ in range(N + 1)]

    if obstacles is None:
        return obs_pred

    for k in range(N + 1):
        t_k = t_now + k * Ts_mpc
        for obst in obstacles:
            pos = obst.position(t=t_k)
            ox, oy = float(pos[0]), float(pos[1])

            # TODO: choose a conservative radius for obstacle
            # Example: if obstacle has size() or radius():
            if hasattr(obst, "radius"):
                r_obs = float(obst.radius())
            elif hasattr(obst, "size"):
                size = obst.size()
                # conservative: half of max XY size
                r_obs = 0.5 * max(float(size[0]), float(size[1]))
            else:
                r_obs = 0.5  # fallback

            r_safe = r_obs + robot_radius + margin
            obs_pred[k].append((ox, oy, r_safe))

    return obs_pred



