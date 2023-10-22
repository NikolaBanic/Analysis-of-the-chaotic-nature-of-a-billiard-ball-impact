# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:28:11 2023

@author: nikol
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy.integrate import odeint


def pool(fi_start, fi_stop, n, speed, X_P, Y_P, plot = False, animate = False, analiza = False):
    
    Xb, Yb, S, angle, res = [], [], [], [], []
    count_ = np.zeros([1, 8])
    
    """
    Saving directory
    """
    res_dir = 'pictures'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    """
    Definiranje variabli i veličinu stola
    """
    k = 0.25  # koeficijent otpora zraka
    m = 0.210  # kg
    r = 0.99  # Faktor smanjenja brzine

    # Stol za biljar
    # Unutarnje granice
    LX_I, RX_I = 0, 0.991
    DY_I, UY_I = 0, 1.981
    # Vanjske granice
    LX_O, RX_O = -0.127, 1.118
    DY_O, UY_O = -0.127, 2.108

    X_I = [LX_I, RX_I, RX_I, LX_I, LX_I]
    Y_I = [DY_I, DY_I, UY_I, UY_I, DY_I]
    X_O = [LX_O, RX_O, RX_O, LX_O, LX_O]
    Y_O = [DY_O, DY_O, UY_O, UY_O, DY_O]

    # Scale factor rupa m
    HS = 0.06
    len_k = 45
    # Raspon kuteva
    kut_0_90 = np.linspace(0, np.pi/2, len_k)
    kut_90_180 = np.linspace(np.pi/2, np.pi, len_k)
    kut_180_270 = np.linspace(np.pi, 3/2 * np.pi, len_k)
    kut_270_360 = np.linspace(3/2 * np.pi, 2 * np.pi, len_k)
    kut_180_360 = np.linspace(np.pi, 2 * np.pi, len_k)
    kut_0_180 = np.linspace(0, np.pi, len_k)

    # Rupe
    kut_1X, kut_1Y = LX_I + np.sin(kut_0_90) * HS, DY_I + np.cos(kut_0_90) * HS
    kut_2X, kut_2Y = RX_I + np.sin(kut_270_360) * \
        HS, DY_I + np.cos(kut_270_360) * HS
    kut_3X, kut_3Y = RX_I + \
        np.sin(kut_180_360) * HS/np.sqrt(2), UY_I / \
        2 + np.cos(kut_180_360) * HS/np.sqrt(2)
    kut_4X, kut_4Y = RX_I + np.sin(kut_180_270) * \
        HS, UY_I + np.cos(kut_180_270) * HS
    kut_5X, kut_5Y = LX_I + np.sin(kut_90_180) * HS, UY_I + np.cos(kut_90_180) * HS
    kut_6X, kut_6Y = LX_I + \
        np.sin(kut_0_180) * HS/np.sqrt(2), UY_I / \
        2 + np.cos(kut_0_180) * HS/np.sqrt(2)

    # Početni udarac pozicija
    P_X, P_Y = X_P * RX_I/2, Y_P * UY_I/5
    x0, y0 = X_P * RX_I/2, Y_P * UY_I/5
    
    if plot:
        fig, axes = plt.subplots(
            nrows=1, ncols=1, figsize=(6, 8), constrained_layout=True)
        plt.plot(X_I, Y_I, color='k', lw=1)
        plt.plot(X_O, Y_O, color='k', lw=1)
        plt.plot(P_X, P_Y, color='white', marker='o', markersize=10)
        plt.fill_between(X_O, Y_O, color='yellow')
        plt.fill_between(X_I, Y_I, DY_I, color='green')
        plt.fill_between(kut_1X, kut_1Y, color='red')
        plt.fill_between(kut_2X, kut_2Y, color='red')
        plt.fill_between(kut_3X, kut_3Y, UY_I/2, color='red')
        plt.fill_between(kut_4X, kut_4Y, UY_I, color='red')
        plt.fill_between(kut_5X, kut_5Y, UY_I, color='red')
        plt.fill_between(kut_6X, kut_6Y, UY_I/2, color='red')

        plt.title('Analysis of a Billiard Ball Strike')
        plt.axis('equal')
        plt.ylabel('Billiard Table Height [m]')
        plt.xlabel('Billiard Table Width [m]')

    # for i in range(fi_start, fi_stop, n):
    for i in np.arange(fi_start, fi_stop, n):
        angle.append(i)
        fi = i * (np.pi/180)
        vx0, vy0 = speed * np.cos(fi), speed * np.sin(fi)
        x0, y0 = X_P * RX_I/2, Y_P * UY_I/5

        X, Y = [], []
        tan_v0 = np.sqrt(vx0**2 + vy0**2)
        # while abs(vx0) >= 0.1 or abs(vy0) >= 0.1:
        while abs(tan_v0) >= 0.1:
            def biljar(Z, t):
                x, y, vx, vy = Z
                Dx = vx
                Dy = vy
                Dvx = (-k/m) * Dx
                Dvy = (-k/m) * Dy
                return np.array([Dx, Dy, Dvx, Dvy])
        
            # Početni uvijeti
            Z0 = np.array([x0, y0, vx0, vy0])
            tstop = 10
            nt = 1000
            T = np.linspace(0, tstop, nt)
            R = odeint(biljar, Z0, T)
            x, y, vx, vy = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
            tan_v0 = np.sqrt(vx0**2 + vy0**2)
        
            x1, y1 = [], []
            countx, county = 0, 0
            for i, j in zip(x, y):
                if LX_I <= i <= RX_I:
                    countx += 1
                    x1.append(i)
                if DY_I <= j <= UY_I:
                    county += 1
                    y1.append(j)
        
            if countx < county:
                smjer_x = -1
                smjer_y = 1
            else:
                smjer_x = 1
                smjer_y = -1
        
            bb = min(countx, county)
            x = x[:bb]
            y = y[:bb]
            x0 = x[-1]
            y0 = y[-1]
            vx = vx[:bb]
            vy = vy[:bb]
            vx0 = smjer_x * r * vx[-1]
            vy0 = smjer_y * r * vy[-1]
        
            pogodak = 0
            H = 0
            for i in range(len(x)):
                for j in range(len_k):
                    if x[i] < kut_1X[j] and y[i] < kut_1Y[j]:
                        pogodak = 1
                        if pogodak == 1:
                            H = 1
                            break
                    if x[i] > kut_2X[j] and y[i] < kut_2Y[j]:
                        pogodak = 1
                        if pogodak == 1:
                            H = 2
                            break
                    if x[i] > kut_3X[j] and y[i] > kut_3Y[j] and min(kut_3Y) < y[i] < max(kut_3Y):
                        pogodak = 1
                        if pogodak == 1:
                            H = 3
                            break
                    if x[i] > kut_4X[j] and y[i] > kut_4Y[j]:
                        pogodak = 1
                        if pogodak == 1:
                            H = 4
                            break
                    if x[i] < kut_5X[j] and y[i] > kut_5Y[j]:
                        pogodak = 1
                        if pogodak == 1:
                            H = 5
                            break
                    if x[i] < kut_6X[j] and y[i] < kut_6Y[j] and min(kut_6Y) < y[i] < max(kut_6Y):
                        pogodak = 1
                        if pogodak == 1:
                            H = 6
                            break
        
            xl = x.tolist()
            yl = y.tolist()
            X.append(xl)
            Y.append(yl)
                    
            if pogodak == 1:
                print(f'Shot into pocket number {H}')
                res.append(f'Shot into pocket number {H}')
                break

        if pogodak == 0:
            print('Miss')
            res.append('Miss')
        
        flat_X = [item for sublist in X for item in sublist]
        flat_Y = [item for sublist in Y for item in sublist]
        
        Xb.append(flat_X)
        Yb.append(flat_Y)
        S.append(H)
        
        count_0 = S.count(0)
        count_hit = len(S) - count_0
        count_hit = (count_hit/len(S)) * 100
        count_0 = (count_0/len(S)) * 100
        count_1 = (S.count(1)/len(S)) * 100
        count_2 = (S.count(2)/len(S)) * 100
        count_3 = (S.count(3)/len(S)) * 100
        count_4 = (S.count(4)/len(S)) * 100
        count_5 = (S.count(5)/len(S)) * 100
        count_6 = (S.count(6)/len(S)) * 100
        
        count_[0, 0], count_[0, 1] = count_0, count_hit
        count_[0, 2], count_[0, 3] = count_1, count_2
        count_[0, 4], count_[0, 5] = count_3, count_4
        count_[0, 6], count_[0, 7] = count_5, count_6
        
    if plot:
        if animate:
            

            angle_template = 'Angle = %.f degrees'
            stroke_count_template = 'Strokes = %.f'
            miss_count_template = 'Misses = %.f'
            hit_count_template = 'Hits = %.f'

            kut_text = axes.text(0.70, 0.97, '', transform=axes.transAxes)
            res_text = axes.text(0.05, 0.97, '', transform=axes.transAxes)
            hole1_text = axes.text(0.1, 0.01, '', transform=axes.transAxes)
            hole2_text = axes.text(0.4, 0.01, '', transform=axes.transAxes)
            hole3_text = axes.text(0.7, 0.01, '', transform=axes.transAxes)
            
            line1, = axes.plot([], [], ls = '-',c = 'white',lw = 1,ms = 2)
            line2, = axes.plot([], [], marker = 'o', c = 'black', lw = 1, ms = 10)
        
            def init():
                
                line1.set_data([],[])
                line1.set_data([],[])

                return line1, line2
                
            def animate(i):
                markerx = [Xb[i]]
                markery = [Yb[i]]
                endballx =[Xb[i][-1]]
                endbally = [Yb[i][-1]]
                
                line1.set_data(markerx, markery)    
                line2.set_data(endballx, endbally) 
                kut_text.set_text(angle_template%(angle[i]))      
                res_text.set_text(res[i])

                hole1_text.set_text(stroke_count_template%(len(S)))
                hole2_text.set_text(miss_count_template%(count_0))
                hole3_text.set_text(hit_count_template%(count_hit))

                return line1, line2 
    
            global anim
            anim = animation.FuncAnimation(fig, animate, frames = np.arange(0, len(Xb)),
                                           interval= 500, init_func=init, blit=False, repeat = False)
            anim.save(f'{res_dir}/animation.avi', fps = 10, dpi = 300)
        plt.show()
            

    if analiza:
        fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 8), constrained_layout = True)
        axes[0].plot(angle, S, marker = 'o', mfc = 'orange', mec = 'black', ms = 4)
        axes[0].set_xlabel('Angle of Shot')
        axes[0].set_ylabel('Pocket Label: 1-6, Miss: 0')
        axes[0].set_title(f'Analysis of Shots in the Angle Range: {fi_start}° to {fi_stop}°')
        axes[0].text(0.70, 0.90, f'Initial Shot Speed: {speed} m/s', transform=axes[0].transAxes)
        axes[0].text(0.70, 0.95, f'Angle Interval between Shots: {n}°', transform=axes[0].transAxes)
        lbl1 = axes[1].bar(0, count_0, label = 'Number of Misses', color = 'red')
        lbl2 = axes[1].bar(1, count_hit, label = 'Number of Hits', color = 'green')
        lbl3 = axes[1].bar([2, 3, 4, 5, 6, 7], [count_1, count_2, count_3, count_4, count_5, count_6],
                    label = 'Hit in Pocket Number', color = 'blue')
        axes[1].legend(loc = 'upper right')
        axes[1].set_ylabel('Percentage of Hits/Misses %')
        axes[1].set_title('Hits/Misses by Pocket Label')
        labels_hole = ['Miss', 'Hit', 'Pocket 1', 'Pocket 2', 'Pocket 3', 'Pocket 4', 'Pocket 5', 'Pocket 6']
        x_w = np.arange(len(labels_hole))
        axes[1].set_xticks(x_w, labels = labels_hole)
        axes[1].bar_label(lbl1, fmt = '%.1f%%')
        axes[1].bar_label(lbl2, fmt = '%.1f%%')
        axes[1].bar_label(lbl3, fmt = '%.1f%%')
        fig.suptitle('Analysis of the Chaotic Path of a Billiard Ball')
        
        plt.savefig(f'{res_dir}/{fi_start}_{fi_stop}_{n}_{speed}_shot_analysis.png', dpi = 300)
        plt.show()
    return Xb, Yb, S, angle, res, count_

if __name__ == '__main__': 
    
    """
    Saving directory
    """
    res_dir = 'pictures'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    
    
    # početni kut, krajnji kut udarca, razmak kuteva, brzina udarca
    delta = 1
    fi_start, fi_stop, speed = 0, 90, 7
    X_P, Y_P = 1, 1
    
    A = pool(fi_start, fi_stop, delta, speed, X_P, Y_P, plot = True, animate = True, analiza = True)
    
    
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 8), constrained_layout = True)
    delta = [0.5, 1, 2, 3, 4]
    labels = ['Miss', 'Hit', 'Pocket 1', 'Pocket 2', 'Pocket 3', 'Pocket 4', 'Pocket 5', 'Pocket 6']
    x_width = np.arange(len(labels))
    width = 0.18
    w = [-5/2 * width, -3/2 * width, -1/2 * width, 1/2 * width, 3/2 * width]
    colors = ['green', 'orange', 'purple', 'pink', 'red']
    for i in range(len(delta)):
        fi_start, fi_stop, speed = 0, 90, 7
        X_P, Y_P = 1, 1
        R = pool(fi_start, fi_stop, delta[i], speed, X_P, Y_P, plot = False, animate = False, analiza = False)
        R_i = []
        for j in range(8):
            R_i.append(R[5][0, j])
        bar = axes.bar(x_width + w[i], R_i, width, color = colors[i], label = f'Δ angle = {delta[i]}°', alpha = 0.9)
        if i % 2 == 1:
            axes.bar_label(bar, fmt = '%.1f%%', padding = 4)
        else:
            axes.bar_label(bar, fmt = '%.1f%%', label_type = 'center')
    axes.legend(loc = 'upper right')
    axes.text(0.77, 0.92, f'Initial Shot Speed: {speed} m/s', transform=axes.transAxes)
    axes.set_xticks(x_width, labels)
    axes.set_ylabel('Percentage of Hits/Misses %')
    axes.set_title('Analysis of a Billiard Ball Shot by Δ Angle')
    plt.savefig(f'{res_dir}/shot_analysis_single_plot.png', dpi = 300)
    plt.show()
        
    """
    2D analyis - uncomment - takes a lot of calculations
    """
    # X_P, Y_P = 0, 1
    # X_ = []
    # fi_start, fi_stop, delta, speed = 0, 90, 1, 7
    # pogoci = np.zeros([90, 19])
    # for i in range(19):
    #     X_P += 0.1
    #     X_P = round(X_P, 1)
    #     X_.append(X_P)
    #     A = pool(fi_start, fi_stop, delta, speed, X_P, Y_P, plot = False, animate = False, analiza = False)
    #     pogoci[:, i] = A[2]
    #     print(f'{(i/18)*100}%')
    # x_scale = []
    # for i in range(5,100,5):
    #     x_scale.append(f'{i}%')
    # kutevi = np.arange(fi_start, fi_stop, delta)
    # x_plot, y_plot = np.meshgrid(X_, kutevi)
    # fig, axes = plt.subplots(figsize = (10, 8), constrained_layout = True)
    # contourf_ = axes.pcolormesh(x_plot, y_plot, pogoci, cmap = 'Set2')
    # cbar = fig.colorbar(contourf_, label = 'Pocket Label for Billiard Ball Entry: 1-6; 0 - Miss')
    # axes.set_xticks(X_, x_scale)
    # plt.xlabel('Initial Shot Position along the [%] Width of the Billiard Table (0.991 m)')
    # plt.ylabel('Angle of Shot [°]')
    # plt.title('2D Analysis of Billiard Ball Entry into a Pocket Based on Initial Ball Position')
    # axes.text(0.02, 0.02, f'Initial Shot Speed: {speed} m/s, Delta_angle: {delta}°', transform=axes.transAxes)
    # plt.savefig(f'{res_dir}/shot_analysis.png', dpi = 300)
    # plt.show()
    
    
    # fig, axes = plt.subplots(nrows = 5, ncols = 1, figsize = (16, 8), constrained_layout = True)
    # delta = [0.1, 1, 2, 3, 4]
    # for i in range(len(delta)):
    #     fi_start, fi_stop, speed = 0, 90, 7
    #     X_P, Y_P = 1, 1
    #     R = pool(fi_start, fi_stop, delta[i], speed, X_P, Y_P, plot = False, animate = False, analiza = False)
    #     x_plot = np.array([R[3], R[3]])
    #     y_plot = np.array([np.zeros(len(R[3])), np.ones(len(R[3]))])
    #     r_plot = np.array([R[2], R[2]])
    #     contourf_ = axes[i].pcolormesh(x_plot, y_plot, r_plot, cmap = 'Set2')
    #     x_ = [2, 2, 2, 3, 4]
    #     x_scale = np.arange(fi_start, fi_stop, x_[i])
    #     axes[i].set_xticks(x_scale)
    #     axes[i].set_yticks([])
    #     axes[i].set_ylabel(f'Δ angle = {delta[i]}°')
    #     axes[i].text(0.01, 0.85, f'Speed = {speed} m/s, Δ angle = {delta[i]}°', transform=axes[i].transAxes)
    # plt.suptitle('Analysis of a Billiard Ball Shot by Δ Angle')
    # plt.xlabel('Angle of Shot [°]')
    # cbar = fig.colorbar(contourf_, ax=axes.ravel().tolist(), label = 'Pocket Label for Ball Entry: 1-6; 0 - Miss')
    # plt.savefig(f'{res_dir}/shot_analysis_expanded.png', dpi = 300)
    # plt.show()