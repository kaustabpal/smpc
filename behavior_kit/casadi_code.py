from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from behavior_kit import utils
from behavior_kit.utils import get_dist
import os
import copy
np.set_printoptions(suppress=True)

def DM2Arr(dm):
    return np.array(dm.full())

def draw_circle(x, y, radius):
    th = np.arange(0,2*np.pi,0.01)
    xunit = radius * np.cos(th) + x
    yunit = radius * np.sin(th) + y
    return xunit, yunit  

class Agent:
    def __init__(self, agent_id, i_state, g_state, N=50, obstacles=[]):
        # agent state
        self.sensor_radius = 50
        self.id = agent_id # id of the agent
        self.radius = 1.0
        self.obst_radius = 1.0
        self.i_state = np.array(i_state) # start state
        self.g_state = np.array(g_state) # goal state
        # self.c_state = self.i_state
        self.state_init = ca.DM([i_state[0], i_state[1], i_state[2]])    # initial state
        self.state_target = ca.DM([g_state[0], g_state[1], g_state[2]])  # target state

        # state symbolic variables
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.states = ca.vertcat(
            self.x,
            self.y,
            self.theta
        )
        self.n_states = self.states.numel()

        # control symbolic variables
        self.a = ca.SX.sym('a')
        self.j = ca.SX.sym('j')
        self.controls = ca.vertcat(
            self.a,
            self.j
        )
        self.n_controls = self.controls.numel()

        # self.g_state = np.array(g_state) # goal state
        # self.c_state = i_state # current state
        self.obstacles = []
        self.n_obst = 0
        self.avoid_obs = False
        # horizons
        self.N = N # planning horizon
        self.v_ub = 20
        self.v_lb = 0
        self.w_ub = 0.78
        self.w_lb = -0.78
        self.amin = -5 # -3.19
        self.amax = 5 # 3.19
        self.jmin = -10
        self.jmax = 10
        self.right_lane_bound = 4.5
        self.left_lane_bound = -4.5
        # dt
        self.dt = 0.041
        self.avg_time = []
        # setting matrix_weights' variables
        self.Q_x = 100
        self.Q_y = 100
        self.Q_theta = 500
        # matrix containing all states over all time steps +1 (each column is a state vector)
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        # matrix containing all control actions over all time steps (each column is an action vector)
        self.U = ca.SX.sym('U', self.n_controls, self.N)
        # coloumn vector for storing initial state and target state
        self.P = ca.SX.sym('P', self.n_states + self.n_states)
        # state weights matrix (Q_X, Q_Y, Q_THETA)
        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)

        # Unicycle model
        self.J = ca.vertcat(
            ca.horzcat(cos(self.theta), 0),
            ca.horzcat(sin(self.theta), 0),
            ca.horzcat( 0, 1)
        )
        # RHS = states + J @ controls * self.dt  # Euler discretization
        self.RHS = self.J @ self.controls
        # maps controls from [v, w].T to [vx, vy, omega].T
        self.f = ca.Function('f', [self.states, self.controls], [self.RHS])

        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )

        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)  # initial state full
        self.vl = ca.DM(0) # last velocity
        self.wl = ca.DM(0) # last velocity

    def get_goal_cost(self):
        cost_fn = 0
        for k in range(1,self.N):
            st = self.X[:, k]
            # con = self.U[:, k]
            cost_fn += (st - self.P[self.n_states:]).T @ self.Q @ (st - self.P[self.n_states:])
        return cost_fn
        # st = self.X[:, self.N]
        # return (st - self.P[self.n_states:]).T @ self.Q @ (st - self.P[self.n_states:])
    
    def get_ang_acc_cost(self):
        cost_fn = 0
        for k in range(0,self.N):
            w = self.U[1, k]
            # con = self.U[:, k]
            cost_fn += w**2
        return cost_fn

    def get_lane_cost(self, lane_x):
        cost_fn = 0
        for k in range(self.N+1):
            st_x = self.X[0, k]
            cost_fn += (st_x - lane_x)**2
        # print(cost_fn)
        # quit()
        return cost_fn

    def next_state_constraints(self):
        self.g = self.X[:, 0] - self.P[:self.n_states]  # constraints in the equation
        self.lbg = ca.vertcat(0, 0, 0)
        self.ubg = ca.vertcat(0, 0, 0)
        v1 = self.vl
        w1 = self.wl
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            
            v_ = con[0]
            w_ = con[1]

            v1 += v_*self.dt 
            w1 += w_*self.dt 
            st_next = self.X[:, k+1]

            # k1 = self.f(st, con)
            # k2 = self.f(st + self.dt**2/2*k1, con)
            # k3 = self.f(st + self.dt**2/2*k2, con)
            # k4 = self.f(st + self.dt**2 * k3, con)
            # st_next_RK4 = st + (self.dt**2 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            st_next_update = st
            st_next_update[0,0] = st[0,0] + v1*cos(st[2])*self.dt
            st_next_update[1,0] = st[1,0] + v1*sin(st[2])*self.dt
            st_next_update[2,0] = st[2,0] + w1*self.dt

            self.g = ca.vertcat(self.g, st_next - st_next_update)

            self.lbg = ca.vertcat(self.lbg, 0, 0, 0)  # 0 <= st_next - st_next_RK4 <= 0
            self.ubg = ca.vertcat(self.ubg, 0, 0, 0)  # 0 <= st_next - st_next_RK4 <= 0

    def obstacle_constraints(self):
        for o in self.obstacles:
            dist = np.sqrt((self.state_init[1]-o.state_init[1])**2 + (self.state_init[0]-o.state_init[0])**2)
            if(dist <=self.sensor_radius):
                o_X = DM2Arr(o.X0)
                for i in range(1,self.N+1):
                    a_st = self.X[:,i]
                    o_st = o_X[:,i] #+ np.random.normal(scale = 0.1, size=(3))
                    dist = ca.sqrt((a_st[1]-o_st[1])**2 + (a_st[0]-o_st[0])**2)
                    self.g = ca.vertcat(self.g, dist)
                    self.lbg = ca.vertcat(self.lbg, 2.5)
                    self.ubg = ca.vertcat(self.ubg, ca.inf)

    
    def lane_boundary_constraints(self):
        self.lbx = ca.vertcat( self.left_lane_bound+self.radius, -ca.inf, -ca.inf)
        self.ubx = ca.vertcat( self.right_lane_bound-self.radius, ca.inf, ca.inf)
        for k in range(self.N):
            self.lbx = ca.vertcat(self.lbx, self.left_lane_bound+self.radius, -ca.inf, -ca.inf) # left lane boundary
            self.ubx = ca.vertcat(self.ubx, self.right_lane_bound-self.radius, ca.inf, ca.inf) # right lane boundary

    def control_bound_constraints(self):
        for i in range(self.N):
            self.lbx = ca.vertcat(self.lbx, self.amin, self.jmin)
            self.ubx = ca.vertcat(self.ubx, self.amax, self.jmax)

    def vel_bound_constraints(self):
        v1 = copy.deepcopy(self.vl)
        for i in range(self.N):
            v2 = self.U[0,i]*self.dt + v1
            self.g = ca.vertcat(self.g, v2)
            self.lbg = ca.vertcat(self.lbg, self.v_lb)
            self.ubg = ca.vertcat(self.ubg, self.v_ub)
            v1 = copy.deepcopy(v2)

    def ang_vel_bound_constraints(self):
        w1 = copy.deepcopy(self.wl)
        for i in range(self.N-1):
            w2 = self.U[1,i]*self.dt + w1
            self.g = ca.vertcat(self.g, w2)
            self.lbg = ca.vertcat(self.lbg, self.w_lb)
            self.ubg = ca.vertcat(self.ubg, self.w_ub)
            w1 = copy.deepcopy(w2)

    def pred_controls(self):
        cost_fn = self.get_goal_cost() + self.get_ang_acc_cost()
        # lane_cost = self.get_lane_cost(x_lane)
        cost = cost_fn 
        self.next_state_constraints()
        self.lane_boundary_constraints()
        self.control_bound_constraints()
        self.vel_bound_constraints()
        self.ang_vel_bound_constraints()
        self.obstacle_constraints()

        nlp_prob = {
            'f': cost,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }

        opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        args = {
            'lbg': self.lbg, #ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
            'ubg': self.ubg, #ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound
            'lbx': self.lbx,
            'ubx': self.ubx,
            'p': ca.vertcat(self.state_init, self.state_target ),
            'x0': ca.vertcat(ca.reshape(self.X0, self.n_states*(self.N+1), 1), ca.reshape(self.u0, self.n_controls*self.N, 1))
        }

        t1 = time()
        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        self.avg_time.append(time()-t1)

        self.u0 = ca.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        self.X0 = ca.reshape(sol['x'][: self.n_states * (self.N+1)], self.n_states, self.N+1)
        # print(self.u0)
        # quit()
            
        

if __name__ == '__main__':
    rec_video = False
    exp_name = "Obstacle_Max_Brake"
    exp_num =  exp_name
    os.makedirs(exp_name+"/tmp/", exist_ok=True)
    timeout = 20
    times = np.array([[0]])
    y_lane = np.arange(-1000,1000)
    x1_l_lane = -4*np.ones(y_lane.shape)
    x1_m_lane = -2*np.ones(y_lane.shape)
    x1_r_lane = 0*np.ones(y_lane.shape)

    draw_list = []
    # v_a = []
    y_l_lim = -10
    y_u_lim = 40
    update_y = 0
    a = Agent(1, [-2, 0, np.deg2rad(90)],[-2,40,np.deg2rad(90)], 30)
    b = Agent(2, [-2, 4, np.deg2rad(90)],[-2,44,np.deg2rad(90)], 30)
    a.obstacles = [b]
    draw_list.append(a)
    draw_list.append(b)
    if(rec_video):
        plt_sv_dir = exp_num+"/tmp/"
        p = 0
    x_lane = -2
    while( (ca.norm_2(a.state_init - a.state_target)>=1) and timeout >0):
        timeout = timeout - a.dt
        # t1 = time()
        a.pred_controls()
        a.vl = a.u0[0,0]
        a.wl = a.u0[1,0]
        a.state_init = a.X0[:,1]
        
        b.pred_controls()
        b.vl = b.u0[0,0]
        b.wl = b.u0[1,0]
        b.state_init = b.X0[:,1]
        
        if(b.state_init[1] >=80):
            b.amax = b.amin


        utils.draw(draw_list)
        plt.plot(x1_r_lane,y_lane, 'k', linewidth=1)
        plt.plot(x1_m_lane, y_lane, 'k', linestyle="dotted", linewidth=1)
        plt.plot(x1_l_lane,y_lane, 'k', linewidth=1)

        update_y = update_y + 1
        if(update_y>= 25):
            update_y = 0
            y_l_lim = int(a.state_init[1] - 9)
            y_u_lim = int(a.state_init[1] + 61)
        plt.xlim([-35, 35])
        plt.ylim([y_l_lim, y_u_lim])

        if(rec_video):
            plt.savefig(plt_sv_dir+str(p)+".png",dpi=500, bbox_inches='tight')
            p = p+1
            plt.clf()
        else:
            plt.pause(1e-10)
            plt.clf()
        if(a.g_state[1] <=500):
            a.state_target[1] = a.state_init[1] + 40
        b.state_target[1] = b.state_init[1] + 40


    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
#    if(rec_video):
#        os.system('ffmpeg -r 10 -f image2 -i '+exp_num+'/tmp/%d.png -s 1000x1000 -pix_fmt yuv420p -y '+exp_num+'/'+exp_name+'.mp4')
