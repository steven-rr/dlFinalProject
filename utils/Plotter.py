import matplotlib.pyplot as plt
import os
import constants
import numpy as np

class Plotter:
    @staticmethod
    def format_path(name):
        if not os.path.exists(constants.IMAGE_ROOT): os.makedirs(constants.IMAGE_ROOT)
        return os.path.join(constants.IMAGE_ROOT, name)

    @staticmethod
    def save_plot(name):
        plt.savefig(Plotter.format_path(name) + ".png", bbox_inches='tight')	
        plt.clf()

    @staticmethod
    def get_axis(axis, title='', grid=True):
        if axis is None:
            plt.figure(figsize=(8,8))
            axis = plt.gca()
        if title:   
            axis.set_title(title)
        if grid:
            axis.grid()        
        return axis

    @staticmethod
    def plot_times(x, vi, pi, axis=None, w=None):
        axis = Plotter.get_axis(axis, "Convergenge Wall Clock Times", False)
        # set width of bar
        barWidth = 0.25
        
        # Set position of bar on X axis
        r1 = np.arange(len(vi))
        r2 = [x + barWidth for x in r1]
        
        # Make the plot
        axis.bar(r1, vi, color='r', width=barWidth, edgecolor='white', label='Value Iteration')
        axis.bar(r2, pi, color='b', width=barWidth, edgecolor='white', label='Policy Iteration')
        
        if w is not None:
            r3 = [x + 2*barWidth for x in r1]
            axis.bar(r3, w, color='orange', width=barWidth, edgecolor='white', label='Q Learning')
        
        # Add xticks on the middle of the group bars
        axis.set_xlabel('Lake')
        axis.set_ylabel('Seconds')
        
        axis.set_xticks([i + barWidth + (0 if w is None else (0.125)) for i in range(len(x))])
        axis.set_xticklabels(x)
        
        # Create legend & Show graphic
        axis.legend()

        
    @staticmethod
    def plot_delta(deltas, axis=None, title='Delta by Iteration', label='Delta', plot_label=''):
        axis = Plotter.get_axis(axis, title)
        axis.set_xlabel('Iterations')
        axis.set_ylabel(label)
        axis.grid()
        axis.plot([i+1 for i in range(len(deltas))], deltas, label=plot_label)

    @staticmethod
    def plot_rewards_deltas(rewards, deltas, axis=None, loc='best', title=f'Deltas and Rewards by Iteration'):
        axis = Plotter.get_axis(axis, title)
        x = [i+1 for i in range(len(deltas))]
        
        axis.set_xlabel('Iterations')
        axis.set_ylabel('Delta')
        ln1 = axis.plot(x, deltas, color='tab:red', label='Deltas')
        axis.tick_params(axis='y')

        ax2 = axis.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Rewards')  
        ln2 = ax2.plot(x, rewards, color='tab:blue', label='Rewards')
        ax2.tick_params(axis='y')

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        axis.legend(lns, labs, loc=loc)

    @staticmethod
    def choose_text_size(n_chars, boxsize=1.0):
        return min(40.0, 80.0 / n_chars) * boxsize
    
    @staticmethod
    def plot_env(env, policy, value, ax, edgecolor='k', show_pol=True, show_value=False, title='Environment'):
        f_char_len = lambda x: len(str(x))
        vectorized = np.vectorize(f_char_len)
        size_policy = Plotter.choose_text_size(np.max(vectorized(policy))) * .5
        value = np.around(value, decimals=2)
        f_char_len = lambda x: len(str(x))
        vectorized = np.vectorize(f_char_len)
        size_value = Plotter.choose_text_size(np.max(vectorized(value))) * .5

        rows, cols = env.desc.shape
        for row in range(rows):
            for col in range(cols):
                x, y = Plotter.rc_to_xy(row, col, rows)
                char = env.desc[row, col]
                patch = plt.Rectangle((x, y), width=1, height=1, edgecolor=edgecolor, facecolor=env.color_map[char], linewidth=0.1)
                ax.add_patch(patch)

                if char != b'F' and char != b'S':
                    continue

                x_center = x + 0.5
                y_center = y + 0.5
                index = (row * cols) + col
                action = np.argmax(policy[index])
                if show_pol:
                    ax.text(x_center, y_center + (-0.25 if show_value else 0), env.action_as_char[action], weight='bold', size=size_policy, horizontalalignment='center', verticalalignment='center', color=edgecolor)
                if show_value:
                    ax.text(x_center, y_center + 0.25, str(value[index]), weight='bold', size=size_value, horizontalalignment='center', verticalalignment='center', color=edgecolor)

        ax.set_title(title)
        ax.axis('off')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.get_figure().tight_layout()

        return ax

    @staticmethod
    def rc_to_xy(row, col, rows):        
        x = col
        y = rows - row - 1
        return x, y

    @staticmethod
    def plot_scree(data_provider, axis=None, n_components=None, whiten=False):
        axis = Plotter.get_axis(axis, f'{data_provider.display_name} - Scree Plot', False)
        pca, _ = data_provider.do_pca(n_components=n_components, whiten=whiten)
        per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
        labels = [str(x) for x in range(1, len(per_var)+1)]
        
        axis.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
        axis.set_ylabel('Percentage of Explained Variance')
        axis.set_xlabel('Principal Components')