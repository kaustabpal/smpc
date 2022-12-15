import torch
import numpy as np 
import matplotlib.pyplot as plt

def smoothen(data_list):
    last = data_list[0]  # First value in the plot (first timestep)
    weight = 0.9
    smoothed = list()
    for point in data_list:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def DM2Arr(dm):
    return np.array(dm.full())

def get_dist(a_state, o_state):
    d = np.sqrt((o_state[0] - a_state[0])**2 + (o_state[1] - a_state[1])**2)
    return d      

def draw_circle(x, y, radius):
    th = torch.arange(0,2*np.pi,0.01)
    xunit = radius * torch.cos(th) + x
    yunit = radius * torch.sin(th) + y
    return xunit, yunit  

def draw_balls(balls):
    for i in range(len(balls)):
        center, radius = balls[i]
        b_x, b_y = draw_circle(center[0], center[1], radius)
        plt.plot(b_x, b_y, 'blue', linewidth=1, alpha=0.1)

def draw(agent_list):
    for i in range(len(agent_list)):
        a = agent_list[i]
        # print(a.X0)
        x = DM2Arr(a.X0)
        u = DM2Arr(a.u0)
        g_state = DM2Arr(a.state_target)
        if(a.id == 1):
            col = 'g'
            plt.scatter(g_state[0], g_state[1], marker='x', color='r')
            plt.scatter(x[0,1:],x[1,1:], marker='.', color='blue', s=1.5)
            x_sensor, y_sensor = draw_circle(x[0,1], x[1,1], a.sensor_radius) 
            plt.plot(x_sensor, y_sensor, col, linewidth=1)
            
        else:
            col = 'r'
            plt.scatter(x[0,1:],x[1,1:], marker='.', color=col, s=2)

        x_a, y_a = draw_circle(x[0,1], x[1,1], a.radius) 
        plt.plot(x_a, y_a, col, linewidth=1)


        plt.annotate(str(a.id), xy=(x[0,1]+0.1, x[1,1]+1.2), size=7)
        plt.annotate(str(a.vl), xy=(x[0,1]+0.1, x[1,1]-0.5), size=6)
        

        # plt.plot([(x[0,1]), float(g_state[0])], [x[1,1], float(g_state[1])], linestyle='dotted', c='k')
        

