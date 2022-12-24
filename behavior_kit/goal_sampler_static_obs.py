import torch
import ghalton
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import copy
import time
from behavior_kit import utils
from behavior_kit.utils import get_dist
from scipy.interpolate import BSpline
import scipy.interpolate as si
np.set_printoptions(suppress=True)

class Goal_Sampler:
    def __init__(self, c_state, g_state, vl, wl, obstacles):
        # agent info
        self.balls = []
        self.radius = 1.0
        self.c_state = c_state # start state
        self.g_state = g_state # goal state
        self.step_size_mean = 0.5
        self.step_size_cov = 0.9
        self.avoid_obs = False
        self.vl = vl
        self.wl = wl
        self.v_ub = 20
        self.v_lb = 1
        self.w_ub = 0.4
        self.w_lb = -0.4
        self.max_ctrl = torch.tensor([self.v_ub, self.w_ub])
        self.min_ctrl = torch.tensor([self.v_lb, self.w_lb])
        self.amin = -3.19
        self.amax = 3.19
        self.jmin = -0.1
        self.jmax = 0.1
        self.init_q = [self.vl, self.wl]

        # obstacle info
        self.obst_radius = 1.0
        self.obstacles = obstacles
        self.n_obst = 0
        
        # MPC params
        # self.N = 2 # Number of samples
        self.dt = 0.041
        self.horizon = 30 # Planning horizon
        self.d_action = 2
        self.knot_scale = 4
        self.n_knots = self.horizon//self.knot_scale
        self.ndims = self.n_knots*self.d_action
        self.bspline_degree = 3
        self.num_particles = 100
        self.top_K = int(0.4*self.num_particles) # Number of top samples
        
        self.null_act_frac = 0.01
        self.num_null_particles = round(int(self.null_act_frac * self.num_particles * 1.0))
        self.num_neg_particles = round(int(self.null_act_frac * self.num_particles)) -\
                                                            self.num_null_particles
        self.num_nonzero_particles = self.num_particles - self.num_null_particles -\
                                                            self.num_neg_particles
        self.sample_shape =  self.num_particles - 2

        if(self.num_null_particles > 0):
            self.null_act_seqs = torch.zeros(self.num_null_particles, self.horizon,\
                                                                    self.d_action)
        # self.initialize_mu()
        # self.initialize_sig()

        # Sampling params
        self.perms = ghalton.EA_PERMS[:self.ndims]
        self.sequencer = ghalton.GeneralizedHalton(self.perms)

        # init_q = torch.tensor(self.c_state)
        self.init_action = torch.zeros((self.horizon, self.d_action)) + torch.tensor(self.init_q)
        self.init_mean = self.init_action 
        self.mean_action = self.init_mean.clone()
        self.best_traj = self.mean_action.clone()
        self.init_v_cov = 0.05
        self.init_w_cov = 0.05
        self.init_cov_action = torch.tensor([self.init_v_cov, self.init_w_cov])
        self.cov_action = self.init_cov_action
        self.scale_tril = torch.sqrt(self.cov_action)
        self.full_scale_tril = torch.diag(self.scale_tril)
        
        self.gamma = 0.99
        self.gamma_seq = torch.cumprod(torch.tensor([1]+[self.gamma]*(self.horizon-1)),dim=0).reshape(1,self.horizon)

        self.traj_N = torch.zeros((self.num_particles, self.horizon+1, 3))
        self.controls_N = torch.zeros((self.num_particles, self.horizon, 2))

        self.top_trajs = torch.zeros((self.top_K, self.horizon+1, 3))
        self.top_traj = self.c_state.reshape(1,3)*torch.ones((self.horizon+1, 3))
        
        self.max_free_ball_radius = 2.5
        self.centers = torch.ones(self.horizon+1,2)
        self.free_ball_radius = self.max_free_ball_radius*torch.ones(self.horizon+1,1)

        self.device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        # self.curr_state_N = np.zeros((self.N,1,3))
        # self.V_N_T = np.zeros((self.N, self.horizon))
        # self.W_N_T = np.zeros((self.N, self.horizon))
 
    # def initialize_mu(self): # tensor contain initialized values'''
    #      self.MU = 0*torch.ones((2,self.horizon)) # 2 dim Mu for vel and Angular velocity
    
    # def initialize_sig(self):
    #     self.SIG = 0.7*torch.ones((2,self.horizon))
    
    def bspline(self, c_arr, t_arr=None, n=30, degree=3):
        sample_device = c_arr.device
        sample_dtype = c_arr.dtype
        cv = c_arr.cpu().numpy()
        if(t_arr is None):
            t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
        # else:
        #     t_arr = t_arr.cpu().numpy()
        spl = si.splrep(t_arr, cv, k=degree, s=0.0)
        #spl = BSpline(t, c, k, extrapolate=False)
        xx = np.linspace(0, n, n)
        # print(xx)
        # quit()
        samples = si.splev(xx, spl, ext=3)
        samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
        return samples
    
    def scale_controls(self, act_seq):
        return torch.max(torch.min(act_seq, self.max_ctrl),self.min_ctrl)

    def sample_controls(self):
        uniform_halton_samples = torch.tensor(self.sequencer.get(self.sample_shape)) # samples N control points
        erfinv = torch.erfinv(2 * uniform_halton_samples - 1)
        knot_points = torch.sqrt(torch.tensor([2.0])) * erfinv
        # print(knot_points.shape)
        knot_samples = knot_points.view(self.sample_shape, self.d_action, self.n_knots)
        # print(knot_samples.shape)
        self.samples = torch.zeros((self.sample_shape, self.horizon, self.d_action))
        # print(self.samples.shape)
        for i in range(self.sample_shape):
            for j in range(self.d_action):
                self.samples[i,:,j] = self.bspline(knot_samples[i,j,:],n = self.horizon, \
                                                            degree = self.bspline_degree)
        delta = self.samples
        z_seq = torch.zeros(1,self.horizon,self.d_action)
        delta = torch.cat((delta,z_seq),dim=0)
        scaled_delta = torch.matmul(delta, self.full_scale_tril.float()).view(delta.shape[0],
                                                                    self.horizon,
                                                                    self.d_action)    
        act_seq = self.mean_action.unsqueeze(0) + scaled_delta
        act_seq = self.scale_controls(act_seq)
        

        append_acts = self.best_traj.unsqueeze(0)
        
        # if(self.num_null_particles > 0):
        #     # negative action particles:
        #     neg_action = torch.tensor([self.v_lb, 0]) * self.mean_action.unsqueeze(0)
        #     # print(neg_action)
        #     neg_act_seqs = neg_action.expand(self.num_neg_particles,-1,-1)
        #     append_acts = torch.cat((append_acts, self.null_act_seqs, neg_act_seqs),dim=0)
      
        act_seq = torch.cat((act_seq, append_acts), dim=0)
        self.controls_N = act_seq
        # print(act_seq.shape, self.controls_N.shape)
        # return act_seq
    
    def unicycle_model(self, state, controls):
        a = torch.tensor([
            [torch.cos(state[2]), 0],
            [torch.sin(state[2]), 0],
            [0, 1]
            ],dtype=torch.float32)

        state = state + a@controls.float()*self.dt
        return state
    
    def rollout(self):
        self.goal_region_cost_N = torch.zeros((self.traj_N.shape[0]))
        self.left_lane_bound_cost_N = torch.zeros((self.traj_N.shape[0]))
        self.right_lane_bound_cost_N = torch.zeros((self.traj_N.shape[0]))
        left_lane_bound = -4.5
        right_lane_bound = 4.5
        self.in_balls_cost_N = torch.zeros((self.traj_N.shape[0]))
        self.collision_cost_N = torch.zeros((self.traj_N.shape[0]))
        self.ang_vel_cost_N = torch.zeros((self.controls_N.shape[0]))
        diag_dt = self.dt*torch.ones(self.horizon, self.horizon)
        diag_dt = torch.tril(diag_dt)
        t = []
        t_2 = []
        t_3 = []
        t_4 = []
        for i in range(self.controls_N.shape[0]):
            t1 = time.time()
            self.traj_N[i,0,:] = self.c_state.view(3)
            v = self.controls_N[i,:,0].view(-1,1)
            w = self.controls_N[i,:,1].view(-1,1)
            w_dt = diag_dt@w.float()
            theta_0 = self.traj_N[i,0,2]*torch.ones(self.horizon,1)
            x_0 = self.traj_N[i,0,0]*torch.ones(self.horizon,1)
            y_0 = self.traj_N[i,0,1]*torch.ones(self.horizon,1)
            theta_new = theta_0 + w_dt
            c_theta = torch.cos(theta_new)
            s_theta = torch.sin(theta_new)
            v_cos_dt = (c_theta.squeeze(1)*diag_dt)@v.float()
            v_sin_dt = (s_theta.squeeze(1)*diag_dt)@v.float()
            x_new = x_0 + v_cos_dt
            y_new = y_0 + v_sin_dt
            self.traj_N[i,1:,:] = torch.hstack((x_new, y_new, theta_new))
            t.append(time.time() - t1)  
            
            # lane boundary constraints
            t3 = time.time()
            left_lane_cost = 1000*torch.ones(self.traj_N[i,self.traj_N[i,:,0]<left_lane_bound,0].shape)
            right_lane_cost = 1000*torch.ones(self.traj_N[i,self.traj_N[i,:,0]>right_lane_bound,0].shape)
            self.left_lane_bound_cost_N[i] = torch.sum(left_lane_cost) 
            self.right_lane_bound_cost_N[i] = torch.sum(right_lane_cost) 
            t_3.append(time.time() - t3)
            
            # angular velocity constraints
            self.ang_vel_cost_N[i] = torch.norm(self.controls_N[i,:,1])
            
            
            # Obstacle avoidance
            # t1 = time.time()
            # for o in self.obstacles:
            #     dist = torch.linalg.norm(self.traj_N[i,:,:2]-torch.from_numpy(o)[:2]*torch.ones(self.horizon+1,2),axis = 1)
            #     self.collision_cost_N[i] += torch.sum(500*(dist<=2).type(torch.float32))
            #     # print(self.collision_cost_N[i])
            # t_4.append(time.time()-t1)
            
            # free balls constraints
            for j in range(1,self.controls_N.shape[1]+1):
                # print(j, len(self.balls))
                ## Free balls cost
                t2 = time.time()
                center, radius = self.balls[j-1]
                d = torch.sqrt( (center[0]-self.traj_N[i,j,0])**2 + (center[1]-self.traj_N[i,j,1])**2)-1
                # self.in_balls_cost_N[i] +=d
                if(d>=radius):
                    self.in_balls_cost_N[i] +=500
                t_2.append(time.time()-t2)
            
                    
                # # Lane boundary cost    
                # t3 = time.time()
                # if(self.traj_N[i,j,0] < left_lane_bound):
                #     self.left_lane_bound_cost_N[i] += 50
                # if(self.traj_N[i,j,0]> right_lane_bound):
                #     self.right_lane_bound_cost_N[i] += 50
                # t_3.append(time.time() - t3)
        # print(self.in_balls_cost_N)
        # input()
                    
        t = np.array(t)
        t_2 = np.array(t_2)
        t_3 = np.array(t_3)
        # print("Rollout time: ",np.sum(t))
        # print("Free balls time: ",np.sum(t_2))
        # print("Lane boundary time: ",np.sum(t_3))
        # print("Obstacle avoidance time: ",np.sum(t_4))
        
                                                        
            # radius = 3.5
            # dist = torch.linalg.norm(self.traj_N[i, self.horizon,:2] - self.g_state[:2])
            # if(dist<=radius):
            #     self.goal_region_cost_N[i] = 0
            # else:
            #     self.goal_region_cost_N[i] = copy.deepcopy(dist)
                
        self.total_cost_N = 1*self.left_lane_bound_cost_N + 1*self.right_lane_bound_cost_N + 1*self.in_balls_cost_N + \
            1*self.ang_vel_cost_N + 0*self.collision_cost_N
        top_values, top_idx = torch.topk(self.total_cost_N, self.top_K, largest=False, sorted=True)
        self.top_trajs = torch.index_select(self.traj_N, 0, top_idx)
        top_controls = torch.index_select(self.controls_N, 0, top_idx)
        self.best_traj = copy.deepcopy(top_controls[0,:,:])
        top_cost = torch.index_select(self.total_cost_N, 0, top_idx)
        w = self._exp_util(top_cost)
        return w, top_controls
        
    def grad(self,center,i):
        left_lane_bound = -4.5
        right_lane_bound = 4.5
        dx = [0.1, 0.0]
        dy = [0.0, 0.1]
        min_dx_plus = 9999999999
        min_dx_minus = 9999999999
        min_dy_plus = 9999999999
        min_dy_minus = 9999999999
        for o in self.obstacles:
            dist_dx_plus = torch.sqrt((center[1]-o[1])**2 + (center[0]+dx[0]-o[0])**2)-2
            dist_dx_minus = torch.sqrt((center[1]-o[1])**2 + (center[0]-dx[0]-o[0])**2)-2
            
            dist_dy_plus = torch.sqrt((center[1]+dy[1]-o[1])**2 + (center[0]-o[0])**2)-2
            dist_dy_minus = torch.sqrt((center[1]-dy[1]-o[1])**2 + (center[0]-o[0])**2)-2
            
            if(dist_dx_plus<min_dx_plus):
                min_dx_plus = dist_dx_plus
                
            if(dist_dx_minus<min_dx_minus):
                min_dx_minus = dist_dx_minus
                
            if(dist_dy_plus<min_dy_plus):
                min_dy_plus = dist_dy_plus
                
            if(dist_dy_minus<min_dy_minus):
                min_dy_minus = dist_dy_minus
                
        dist_dx_plus_left_lane = torch.sqrt((center[0]+dx[0]-left_lane_bound)**2)
        dist_dx_minus_left_lane = torch.sqrt((center[0]-dx[0]-left_lane_bound)**2)
        
        dist_dx_plus_right_lane = torch.sqrt((center[0]+dx[0]-right_lane_bound)**2)
        dist_dx_minus_right_lane = torch.sqrt((center[0]-dx[0]-right_lane_bound)**2)
        
        if(dist_dx_plus_left_lane<min_dx_plus):
            min_dx_plus = dist_dx_plus_left_lane
            
        if(dist_dx_plus_right_lane<min_dx_plus):
            min_dx_plus = dist_dx_plus_right_lane
            
        if(dist_dx_minus_left_lane<min_dx_minus):
            min_dx_minus = dist_dx_minus_left_lane
            
        if(dist_dx_minus_right_lane<min_dx_minus):
            min_dx_minus = dist_dx_minus_right_lane
        
        grad_vec = torch.tensor([min_dx_plus - min_dx_minus, min_dy_plus - min_dy_minus])
        grad_norm = torch.linalg.norm(grad_vec)
        grad_normalized = grad_vec/grad_norm

        return grad_normalized
    
    def get_free_balls(self):
        balls = []
        left_lane_bound = -4.5
        right_lane_bound = 4.5
        # self.centers[:-1,:] = self.centers[1:,:].clone()
        # self.centers[-1,:] = self.top_trajs[-1,-1,:2].clone()
        # print(self.centers.shape)
        for i in range(1,self.horizon+1):
            # self.centers[i,:] = copy.deepcopy(self.traj_N[-1,i,:2])
            # center = self.centers[i,:]
            min_d = 9999999999
            for o in self.obstacles:
                dist = torch.sqrt((self.centers[i,1]-o[1])**2 + (self.centers[i,0]-o[0])**2)-1
                if(dist<min_d):
                    min_d = copy.deepcopy(dist)
            dist_2_right_lane = torch.sqrt((self.centers[i,0]-right_lane_bound)**2)
            dist_2_left_lane = torch.sqrt((self.centers[i,0]-left_lane_bound)**2)
            if(dist_2_right_lane<=dist_2_left_lane):
                min_lane_dist = dist_2_right_lane
            else:
                min_lane_dist = dist_2_left_lane
            if(min_lane_dist<min_d):
                min_d = copy.deepcopy(min_lane_dist)
                
            if(min_d > self.max_free_ball_radius):
                # self.top_trajs[0,i,0] = copy.deepcopy(center[0])
                # self.top_trajs[0,i,1] = copy.deepcopy(center[1])
                self.free_ball_radius[i] = self.max_free_ball_radius
                balls.append([self.centers[i,:], self.max_free_ball_radius])
            else:   
                # print("True")
                # print(min_d)
                new_center = copy.deepcopy(self.centers[i,:])
                step = 0.0
                current_d = copy.deepcopy(min_d)
                self.step_size = 0.001
                # grad_center = self.grad(self.centers[i,:],i)
                # for k in range(3):
                iter = 0
                while(np.fabs(current_d - (step + min_d)) <=0.005 ):
                    iter += 1
                    print(iter)
                    # print(self.centers[i,:].shape)
                    # print(self.grad(self.centers[i,:]).shape)
                    # quit()
                    new_center = self.centers[i,:] + (self.grad(self.centers[i,:],i) * step);
                    # new_min_d = copy.deepcopy(self.max_free_ball_radius)
                    for o in self.obstacles:
                        dist = torch.sqrt((new_center[1]-o[1])**2 + (new_center[0]-o[0])**2)-1
                        if(dist<current_d):
                            current_d = copy.deepcopy(dist)
                    dist_2_right_lane = torch.sqrt((new_center[0]-right_lane_bound)**2)
                    dist_2_left_lane = torch.sqrt((new_center[0]-left_lane_bound)**2)
                    if(dist_2_right_lane<=dist_2_left_lane):
                        min_lane_dist = dist_2_right_lane
                    else:
                        min_lane_dist = dist_2_left_lane
                    if(min_lane_dist<current_d):
                        current_d = copy.deepcopy(min_lane_dist)
                    # current_d = copy.deepcopy(new_min_d)
                    step += self.step_size

                # self.top_trajs[0,i,0] = copy.deepcopy(new_center[0])
                # self.top_trajs[0,i,1] = copy.deepcopy(new_center[1])
                self.centers[i,:] = copy.deepcopy(new_center)
                self.free_ball_radius[i] = copy.deepcopy(current_d)
                balls.append([new_center, current_d])
        self.balls = balls    
    
    def _exp_util(self, costs):
        """
            Calculate weights using exponential utility
        """
        beta = 0.99
        # cost_seq = costs  # discounted cost sequence
        # cost_seq = torch.fliplr(torch.cumsum(torch.fliplr(cost_seq), axis=-1))  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
        # cost_seq /= self.gamma_seq  # un-scale it to get true discounted cost to go
        traj_costs = costs

        # traj_costs = torch.sum(traj_costs,1)
        # self.total_cost_N = traj_costs
        # #calculate soft-max
        w = torch.softmax((-1.0/beta) * traj_costs, dim=0)
        return w
       
    def update_distribution(self, top_w, top_controls):
        
        weighted_seq = top_w.to(self.device) * top_controls.to(self.device).T        
        sum_seq = torch.sum(weighted_seq.T, dim=0)

        new_mean = sum_seq
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean
        
        delta = top_controls - self.mean_action.unsqueeze(0)

        weighted_delta = top_w * (delta ** 2).T
        # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
        cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
        self.cov_action = (1.0 - self.step_size_cov) * self.cov_action +\
                self.step_size_cov * cov_update
    
    def plan_traj(self):
        t1 = time.time()
        # self.centers[:,:] = copy.deepcopy(self.top_trajs[0,:,:2])
        # self.get_free_balls()
        # print("Free Balls: ", time.time() - t1)
        # top_w, top_controls = self.get_cost()
        self.cov_action = self.init_cov_action
        # self.scale_tril = torch.sqrt(self.cov_action)
        # self.full_scale_tril = torch.diag(self.scale_tril)
        for i in range(1):
            self.scale_tril = torch.sqrt(self.cov_action)
            self.full_scale_tril = torch.diag(self.scale_tril)
            t1 = time.time()
            self.get_free_balls()
            self.sample_controls()
            # print("Sample Controls: ", time.time() - t1)
            
            t1 = time.time()
            top_w, top_controls = self.rollout()
            # print("Rollout: ", time.time() - t1)
            t1 = time.time()
            self.update_distribution(top_w, top_controls)
            # print("Update_Distribution: ", time.time() - t1)
            # print("#######################")
        self.scale_tril = torch.sqrt(self.cov_action)
        self.full_scale_tril = torch.diag(self.scale_tril)
        self.sample_controls()
        top_w, top_controls = self.rollout()
        # self.centers[:,:] = copy.deepcopy(self.top_trajs[0,:,:2])
        # self.get_free_balls()
        self.mean_action[:-1,:] = self.mean_action[1:,:].clone()
        self.mean_action[-1,:] = self.init_mean[-1,:].clone()
            
        
    
    def get_vel(self, u):
        v1 = self.vl
        w1 = self.wl
        v = torch.zeros(u.shape)
        for i in range(u.shape[1]):
            v[0,i] = v1 + u[0,i]*self.dt
            v1 = v[0,i]
            v[1,i] = w1 + u[0,i]*self.dt
            w1 = v[1,i]       
        return v
                
    
# if __name__ == '__main__':
#     dtype = torch.float32
#     c_state = torch.tensor([-6, 0, np.deg2rad(90)], dtype=dtype).view(3,1)
#     g_state = torch.tensor([-6, 50, np.deg2rad(90)], dtype=dtype).view(3,1)
#     vl = 5
#     wl = 0
#     obs1 = np.array([-6,30, np.deg2rad(90)])
#     obs2 = np.array([-2,10, np.deg2rad(90)])
#     obs3 = np.array([-2,14, np.deg2rad(90)])
#     obstacles = [obs1] #, obs2, obs3]
#     y_lane = np.arange(-1000,1000)
#     x1_l_lane = -4*np.ones(y_lane.shape)
#     x1_m_lane = -2*np.ones(y_lane.shape)
#     x1_r_lane = 0*np.ones(y_lane.shape)
#     x2_m_lane = -6*np.ones(y_lane.shape)
#     x2_l_lane = -8*np.ones(y_lane.shape)
    
#     sampler = Goal_Sampler(c_state, g_state, vl, wl, obstacles=obstacles)
    
#     top_w, top_controls = sampler.get_cost()
#     for k in range(550):
#         plt.plot(x1_r_lane,y_lane, 'k', linewidth=1)
#         plt.plot(x1_m_lane, y_lane, 'k', linestyle="dotted", linewidth=1)
#         plt.plot(x1_l_lane,y_lane, 'k', linewidth=1)
#         plt.plot(x2_m_lane, y_lane, 'k', linestyle="dotted", linewidth=1)
#         plt.plot(x2_l_lane,y_lane, 'k', linewidth=1)
#         plt.plot(obs1[0], obs1[1],'or')
#         t1 = time.time()
#         sampler.cov_action = sampler.init_cov_action
#         sampler.scale_tril = torch.sqrt(sampler.cov_action)
#         sampler.full_scale_tril = torch.diag(sampler.scale_tril)
#         for i in range(2):
#             sampler.update_distribution(top_w, top_controls)
#             sampler.scale_tril = torch.sqrt(sampler.cov_action)
#             sampler.full_scale_tril = torch.diag(sampler.scale_tril)
#             sampler.sample_controls()
#             sampler.rollout()
#             top_w, top_controls = sampler.get_cost()
#         # print(time.time()-t1)
#         # print(k, sampler.traj_N.shape)
#         for j in range(sampler.traj_N.shape[0]):
#                 plt.plot(sampler.traj_N[j,:,0], sampler.traj_N[j,:,1],'.-', alpha=0.05)
#         plt.plot(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], '.-r')
#         utils.draw_balls(sampler.balls)
#         plt.plot(g_state[0,0],g_state[1,0],'rx')
#         plt.xlim([-35, 35])
#         plt.ylim([-10, 40])
#         plt.title(str(k))
#         # plt.ylim([-0.7, 0.7])
#         plt.pause(1/120)   
#         plt.clf() 
#         sampler.c_state = sampler.top_trajs[0,1,:]
#         # print(sampler.top_trajs[0,:,:])
#         sampler.mean_action[:-1,:] = sampler.mean_action[1:,:].clone()
#         sampler.mean_action[-1,:] = sampler.init_mean[-1,:].clone()
        
        
    