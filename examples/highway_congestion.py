import torch
from behavior_kit.goal_sampler import Goal_Sampler
from behavior_kit.casadi_code import Agent
import os
import numpy as np
import time
import casadi as ca
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from behavior_kit import utils
# from utils import get_dist
import copy
np.set_printoptions(suppress=True)

if __name__ == "__main__":
    
    dtype = torch.float32
    rec_video = False
    exp_name = "Goal_Sampling_acc_model_free_balls"
    exp_num =  exp_name
    os.makedirs(exp_name+"/tmp/", exist_ok=True)
    timeout = 30
    times = np.array([[0]])

    agent_v_ub = 20
    agent_v_lb = 0
    # agent_w_ub = 0.1
    # agent_w_lb = -0.1

    y_lane = np.arange(-1000,1000)
    x1_l_lane = 1.5*np.ones(y_lane.shape)
    x1_r_lane = 4.5*np.ones(y_lane.shape)
    x2_l_lane = -1.5*np.ones(y_lane.shape)
    x3_l_lane = -4.5*np.ones(y_lane.shape)

    lane_centers = [-3,0,3]
    min_d = 9999999 # samllest distance to closest lane center
    nearest_lane_cntr = 0 # nearest lane center

    draw_list = []

    y_l_lim = -10
    y_u_lim = 40
    update_y = 30

    agent1 = Agent(1, [3,0,np.deg2rad(90)],[0,0+30,np.deg2rad(90)], 30)
    draw_list.append(agent1)

    obstacles = []
    oy = 0
    obs_x = [-3,0,3]
    obs_y = [15,20,30]
    ox = [3,-3,0,3,0,0,-3,-3,0,3]
    oy = [18,32,50,63,78,89,105,115,109,129]
    for i in range(10):
        obstacles.append(Agent(i+2,[ox[i],oy[i],np.deg2rad(90)],[ox[i],oy[i]+30,np.deg2rad(90)], 30))

    agent1.v_ub = agent_v_ub
    agent1.v_lb = agent_v_lb 
    # agent1.w_lb = agent_w_lb
    # agent1.w_ub = agent_w_ub
    agent1.vl = ca.DM(13)

    o_v_ub = [13,14,13,12,11,13,13,13,13,12]
    for i in range(len(obstacles)):
        obstacles[i].v_ub = 10 #o_v_ub[i] #np.random.randint(11,15)
        obstacles[i].v_lb = 0
        obstacles[i].w_ub = 0
        obstacles[i].w_lb = 0
        obstacles[i].vl = ca.DM(np.random.randint(10,13)) 
        agent1.obstacles.append(obstacles[i])
        draw_list.append(obstacles[i])

    for o in obstacles:
        eo_id = o.id
        o.avoid_obs = True
        for oo in obstacles:
            if(eo_id == oo.id):
                continue
            else:
                o.obstacles.append(oo)

    agent1.avoid_obs = True

    g_region_cntr =  torch.tensor([0,30,np.deg2rad(90)], dtype=dtype)
    agent1.i_state = torch.tensor(agent1.state_init.full(),dtype=dtype).reshape(3)

    ######################
    obs_pos = []
    for o in agent1.obstacles:
        # o.v_ub = 10
        dist = np.sqrt((agent1.state_init[1]-o.state_init[1])**2 + (agent1.state_init[0]-o.state_init[0])**2)
        if(dist <=agent1.sensor_radius):
            obs_pos.append(np.array(o.state_init.full()).reshape(3))
            
    sampler = Goal_Sampler(agent1.i_state, g_region_cntr, agent1.vl.full()[0][0], 0, obstacles=obs_pos)

    agent1.pred_controls()
    sampler.centers = torch.tensor(agent1.X0.full()).T[:,:2]
    # print(sampler.centers.shape)
    # print(torch.tensor(agent1.X0.full()).T.shape)
    # quit()

    sampler.plan_traj()

    agent1_goal = sampler.top_trajs[0,-1,:]
    # print(agent1_goal[1])
    # quit()
    min_d = 999999999
    for d in lane_centers:
        dist = np.sqrt((d-agent1_goal[0])**2)
        if(dist<min_d):
            min_d = dist
            nearest_lane_cntr = d


    agent1.state_target[0] = nearest_lane_cntr # agent1_goal[0]
    agent1.state_target[1] = agent1_goal[1].numpy()

    if(rec_video):
        plt_sv_dir = exp_num+"/tmp/"
        p = 0
    # x_lane = -6
    t_taken = 0
    delta_t = 0
    while( (ca.norm_2(agent1.state_init - agent1.state_target)>=1) and timeout >0):
        timeout = timeout - agent1.dt
        # t1 = time()
        obs_pos = []
        for o in agent1.obstacles:
            dist = np.sqrt((agent1.state_init[1]-o.state_init[1])**2 + (agent1.state_init[0]-o.state_init[0])**2)
            if(dist <=agent1.sensor_radius):
                obs_pos.append(np.array(o.state_init.full()).reshape(3))
        sampler.obstacles = obs_pos
        t1 = time.time()
        sampler.plan_traj()
        delta_t = (delta_t + (time.time() - t1))/2
        print(delta_t)
        agent1_goal = sampler.top_trajs[0,-1,:]
        min_d = 999999999
        for d in lane_centers:
            dist = np.sqrt((d-agent1_goal[0])**2)
            if(dist<min_d):
                min_d = dist
                nearest_lane_cntr = d
        # print(nearest_lane_cntr)
        agent1.state_target[0] = copy.deepcopy(nearest_lane_cntr)
        agent1.state_target[1] = copy.deepcopy(agent1_goal[1].numpy())
        # sampler.mean_action[:,0]
        # sampler.top_trajs[0,:,:] = torch.tensor(agent1.X0.full()).T
        
        t1 = time.time()
        agent1.pred_controls()
        print(time.time() - t1)
        print("#################")
        v = sampler.get_vel(agent1.u0.full())   
        # sampler.centers = torch.tensor(agent1.X0.full()).T[:,:2]
        # sampler.best_traj = sampler.mean_action
        # print(v.T.shape)
        sampler.best_traj = v.T
        
        # t_taken = (t_taken + (time()-t1))/2
        # print(t_taken)
        # a_u1 = agent1.u0
        # a_x1 = agent1.X0

        for o in agent1.obstacles:
            o.pred_controls()

        # v_a.append(u[0,0])
        agent1.vl += agent1.u0[0,0]*agent1.dt
        agent1.wl += agent1.u0[1,0]*agent1.dt
        agent1.state_init = agent1.X0[:,1]
        # print(str(agent1.vl))
        # print(agent1.state_init)
        agent1.i_state = torch.tensor(agent1.state_init.full(),dtype=dtype).reshape(3)
        sampler.c_state = agent1.i_state

        for o in obstacles:
            o.vl += o.u0[0,0]*o.dt
            o.wl += o.u0[1,0]*o.dt
            o.state_init = o.X0[:,1]
            o.i_state = np.array(o.state_init.full()).reshape(3)
            

        utils.draw(draw_list)
        utils.draw_balls(sampler.balls)
        # plt.plot(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], '.-r')
        for j in range(sampler.traj_N.shape[0]):
            plt.plot(sampler.traj_N[j,:,0], sampler.traj_N[j,:,1], alpha=0.09)
        plt.scatter(sampler.top_trajs[0,:,0], sampler.top_trajs[0,:,1], color='green', s=1.5)
        plt.plot(x1_r_lane,y_lane,'k', linewidth=1)
        plt.plot(x1_l_lane,y_lane,'k', linewidth=1)
        plt.plot(x2_l_lane,y_lane,'k', linewidth=1)
        plt.plot(x3_l_lane,y_lane,'k', linewidth=1)

        update_y = update_y + 1
        if(update_y>= 30):
            update_y = 0
            y_l_lim = agent1.i_state[1] - 10
            y_u_lim = agent1.i_state[1] + 40
        plt.xlim([-25,25])
        plt.ylim([y_l_lim, y_u_lim])

        if(rec_video):
            plt.savefig(plt_sv_dir+str(p)+".png",dpi=500, bbox_inches='tight')
            p = p+1
            plt.clf()
        else:
            plt.pause(1e-10)
            plt.clf()

        # if(agent1.state_target[1] <=500):
        #     agent1.state_target[1] = agent1.state_init[1] + 30
            # sampler.g_state[0] = nearest_lane_cntr 
            # sampler.g_state[1] = sampler.c_state[1] + 30
        for o in agent1.obstacles:
            o.state_target[1] = o.state_init[1]+30

    
    print('avg iteration time: ', t_taken, 'sec') #np.array(times).mean() * 1000, 'ms')
    if(rec_video):
        os.system('ffmpeg -r 10 -f image2 -i '+exp_num+'/tmp/%d.png -s 1000x1000 -pix_fmt yuv420p -y '+exp_num+'/'+exp_name+'.mp4')

