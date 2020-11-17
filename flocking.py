import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import time
import math

class Flock:
    def __init__(self, inter_agent_dist = 7, num_of_agents = 10, init_pos = None, init_vel = None, time_step = 0.01, gamma_cq = 1, gamma_cp = 1):
        self.N = num_of_agents
        self.Q = init_pos
        self.P = init_vel
        self.p = np.array([2, 0])
        self.q = np.array([100, 0])
        self.cp = gamma_cp
        self.cq = gamma_cq
        self.dt = time_step
        self.d = inter_agent_dist
        self.r = 0.2*self.d
        self.G = self.get_net(self.Q)
    
    def run_sim(self, T= 10, to_agreement = True):
        self.Q_sim = []
        self.P_sim = []
        self.sim_time = []
        self.p_sim = []
        self.q_sim = []
        
        t = 0
        start = time.time()
        while t < T:
            self.G = self.get_net(self.Q)
            Q = self.Q.copy()
            P = self.P.copy()
            self.Q = self.Q + self.P*self.dt
            self.P = self.P + self.f_gamma(Q, P)*self.dt
            self.q = self.q + self.p*self.dt
            self.sim_time.append(time.time() - start)
            self.Q_sim.append(self.Q)
            self.P_sim.append(self.P)
            self.p_sim.append(self.p)
            self.q_sim.append(self.q)
            if(to_agreement and self.velocity_angle_agreement(self.P)):
                t = T
            elif(to_agreement):
                t += self.dt
                T += self.dt
            else:
                t += self.dt    
    
    def plot_time_series(self, time_step_plot, with_labels = False, node_size = 25, width = 2.5, arrow_width = 0.5):
       
        unit = np.zeros(self.P_sim[time_step_plot].shape)
        norms = np.zeros(self.N)
        for i in range(0, self.N):
            norms[i] = np.linalg.norm(self.P_sim[time_step_plot][i])
            unit[i] = self.P_sim[time_step_plot][i]/norms[i]
        rel = np.zeros(self.P_sim[time_step_plot].shape)
        for i in range(0, self.N):
            rel[i] = unit[i]*(np.linalg.norm(self.P_sim[time_step_plot][i])/max(norms))
        
        for i in range(0, self.N):
            plt.arrow(self.Q_sim[time_step_plot][i,0], self.Q_sim[time_step_plot][i,1],2*rel[i, 0], 2*rel[i, 1], width = arrow_width)
        self.G = self.get_net(self.Q_sim[time_step_plot])
        G = self.G.copy()
        Q = self.Q_sim[time_step_plot].copy()
        node_colors = ['red']
        rel_gamma = self.p_sim[time_step_plot]/max(norms)
        plt.arrow(self.q_sim[time_step_plot][0], self.q_sim[time_step_plot][1],2*rel_gamma[0], 2*rel_gamma[1], width = arrow_width, edgecolor = 'blue',facecolor = 'blue')
        G.add_node(self.N)
        node_colors = []
        for node in G:
            if node < self.N:
                node_colors.append('red')
            elif node == self.N:
                node_colors.append('red')
        Q = np.append(Q, self.q_sim[time_step_plot]).reshape(self.N + 1, 2)
        nx.draw_networkx(G, pos = Q, node_color = node_colors, width = width, node_size = node_size, with_labels = False)
        plt.xlim(plt.xlim()[0]-np.abs(unit[0,0]), plt.xlim()[1] + np.abs(unit[0,0]))
        plt.ylim(plt.ylim()[0]-np.abs(unit[0,1]), plt.ylim()[1] + np.abs(unit[0,1]))
        plt.title("t = %ss" %(time_step_plot/100), fontsize=25)

        return plt
        
    def get_net(self, Q):
        G = nx.Graph()
        for i in range(0, self.N):
            for j in range(0, self.N):
                if(np.linalg.norm(Q[i, :] - Q[j, :]) < self.r):
                    if (not G.has_edge(i, j)):
                        G.add_edge(i, j)
        return (G)
    
    '''navigational feedback eqn 24 '''
    def f_gamma(self, Q, P):
        '''
        self.q and self.p are the states of gamma agent
        '''
        return (-1)*self.cq*(Q - self.q) - self.cp*(P - self.p)
    
    def velocity_angle_agreement(self, v, tol = 1e-6):
        unit = np.zeros(v.shape)
        for i in range(0, len(v)):
            unit[i] = v[i]/np.linalg.norm(v[i])
        ref = np.angle(complex(unit[1, 0], unit[1,1]))
        for i in range(0, len(unit)):
            if(np.abs(ref - np.angle(complex(unit[i, 0], unit[i, 1]))) > tol):
                return False
        return True


if __name__ == "__main__":
    N = 10
    np.random.seed(10)
    Q = np.sqrt(2500)*np.random.randn(N, 2)
    P = 2*np.random.randn(N, 2) - 1
    time_period = 10

    plot1 = plt.figure(1)
    FS = Flock(num_of_agents = N, init_pos = Q, init_vel = P)
    FS.run_sim(T = time_period)
    FS.plot_time_series(0)
    plt.show()
    plot3 = plt.figure(3)
    FS.plot_time_series(100)
    plt.show()

    plot4 = plt.figure(4)
    ims = []
    for i in range(0,101):
      ims.append((FS.plot_time_series(i),))
    im_ani = animation.ArtistAnimation(plot4, ims, interval=1000, repeat_delay=0,blit=True)
    plt.show()

    vel_sim = FS.P_sim
    temp_x = []
    temp_y = []
    vel_agent = []
    for i in range(0, 10):
        for j in range(0, 1000):
            temp_x.append(vel_sim[j][i][0])
            temp_y.append(vel_sim[j][i][1])
            if vel_sim[j][i][0] < 0 or vel_sim[j][i][1] < 0:
                vel_agent.append(-1*math.sqrt(vel_sim[j][i][0]**2 + vel_sim[j][i][1]**2))
            else:
                vel_agent.append(math.sqrt(vel_sim[j][i][0]**2 + vel_sim[j][i][1]**2))
    plot2 = plt.figure(2)
    time_range = np.arange(0, 1000)

    plt.plot(time_range, vel_agent[0:1000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[1000:2000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[2000:3000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[3000:4000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[4000:5000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[5000:6000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[6000:7000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[7000:8000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[8000:9000], linewidth = 0.5)
    plt.plot(time_range, vel_agent[9000:10000], linewidth = 0.5)
    plt.show()       
