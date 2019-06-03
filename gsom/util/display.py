"""
Python display plots:
https://python-graph-gallery.com/all-charts/
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors
from collections import Counter
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import squarify
import os

# root_folder = ''


class Display:

    def __init__(self, gsom_node_map, aggregated_node_map):
        self.gsom_node_map = gsom_node_map
        self.aggregated_node_map = aggregated_node_map

    def setup_labels_for_aggregated_nodemap(self, labels, figure_id, title, output_filename):

        plt.figure(figure_id)
        plt.title(title)

        display_sizes = []
        display_labels = []

        for node in self.aggregated_node_map:

            display_sizes.append(len(node.input_vector_weights))

            lbl_str = ''
            for idx, input_vector_weight in enumerate(node.input_vector_weights):

                if idx % 3 != 0:
                    lbl_str += str(labels[input_vector_weight.weight_label]) + ','
                else:
                    lbl_str += '\n' + str(labels[input_vector_weight.weight_label]) + ','
            display_labels.append(lbl_str)

        squarify.plot(sizes=display_sizes, label=display_labels, alpha=.6)
        plt.axis('off')
        plt.savefig(output_filename + '.jpeg', dpi=1200)

    def setup_labels_for_gsom_nodemap(self, labels, figure_id, title, output_filename):

        plt.figure(figure_id)
        plt.title(title)

        max_count = max([node.get_hit_count() for _, node in self.gsom_node_map.items()])
        listed_color_map = Display._get_color_map(max_count, alpha=0.9)

        for key, value in self.gsom_node_map.items():

            key_split = key.split(':')
            x = int(key_split[0])
            y = int(key_split[1])

            if value.get_hit_count() > 0:
                plt.plot(x, y, 'o', color=listed_color_map.colors[value.get_hit_count()], markersize=2)
                label_str = ','.join([str(labels[lbl_id]) for lbl_id in value.get_mapped_labels()])
                plt.text(x, y + 0.3, label_str, fontsize=4)
            else:
                plt.plot(x, y, 'o', color=listed_color_map.colors[value.get_hit_count()], markersize=2)

        plt.savefig(output_filename + '.jpeg', dpi=1200)

    def setup_heatmap_for_gsom_nodemap_adl_activity(self, labels, title, output_filename):

        activities = self._get_adl_activity_list()

        counter = 1
        for activity_key, activity_val in activities.items():

            plt.figure(counter)
            plt.title(title + ' ' + activity_val)

            # Get the max count for colour pallet
            max_count = 0
            for key, value in self.gsom_node_map.items():
                key_count = len([str(labels[lbl_id]) for lbl_id in value.get_mapped_labels() if
                              labels[lbl_id] == activity_key])
                if key_count > max_count:
                    max_count = key_count

            listed_color_map = Display._get_color_map(max_count, alpha=0.9)

            # Plot the map
            for key, value in self.gsom_node_map.items():

                key_split = key.split(':')
                x = int(key_split[0])
                y = int(key_split[1])

                if value.get_hit_count() > 0:
                    label_list = [str(labels[lbl_id]) for lbl_id in value.get_mapped_labels() if labels[lbl_id] == activity_key]
                    plt.plot(x, y, 'o', color=listed_color_map.colors[len(label_list)], markersize=2)
                    hit_count = len(list(label_list))
                    if hit_count > 0:
                        plt.text(x, y + 0.3, str(hit_count), fontsize=4)
                else:
                    plt.plot(x, y, 'o', color='#FCFCFC', markersize=2)  # grey colour for untapped neurons

            plt.savefig(output_filename + '_' + activity_key + '.jpeg', dpi=1200)
            plt.clf()
            counter += 1

    def setup_labels_for_gsom_nodemap_adl_activity(self, labels, figure_id, title, output_filename):

        plt.figure(figure_id)
        plt.title(title)

        colours, legend_patches = self._get_color_map_adl_activity()

        plt.legend(handles=legend_patches, loc=1, prop={'size': 4})

        for key, value in self.gsom_node_map.items():

            key_split = key.split(':')
            x = int(key_split[0])
            y = int(key_split[1])

            if value.get_hit_count() > 0:
                label_list = [str(labels[lbl_id]) for lbl_id in value.get_mapped_labels()]
                #TODO: Select only labels that have >80%
                most_common_label_str, num_most_common = Counter(label_list).most_common(1)[0]
                label_str = ','.join(list(set(label_list)))

                plt.plot(x, y, 'o', color=colours[most_common_label_str], markersize=2)
                plt.text(x, y + 0.3, label_str, fontsize=4)
            else:
                plt.plot(x, y, 'o', color='#DCDCDC', markersize=2)  # grey colour for untapped neurons

        plt.savefig(output_filename + '.jpeg', dpi=1200)

    def plot_gsom_learning(self, new_node_map, labels, figure_id, title, output_filename):

        self.gsom_node_map = new_node_map

        plt.figure(figure_id)
        plt.title(title)

        # plt.xlim(-50, 50)
        # plt.ylim(-50, 50)

        # colours, legend_patches = self._get_color_map_adl_activity()
        # colours, legend_patches = self._get_color_map_ped_behaviour()
        colours, legend_patches = self._get_color_map_learning(len(set(labels)))

        plt.legend(handles=legend_patches, loc=1, prop={'size': 4})

        for key, value in self.gsom_node_map.items():

            key_split = key.split(':')
            x = int(key_split[0])
            y = int(key_split[1])

            if value.get_hit_count() > 0:
                label_list = [str(labels[lbl_id]) for lbl_id in value.get_mapped_labels()]

                lbl_keys = list(Counter(label_list).keys())
                lbl_perc = [i / len(label_list) for i in list(Counter(label_list).values())]

                # Select only labels that have >=50%
                most_common_labels = [x for x, y in sorted(zip(lbl_keys, lbl_perc), key=lambda pair: pair[1]) if y >= 0.3]

                if len(most_common_labels) > 0:
                    plt.plot(x, y, 'o', color=colours[most_common_labels[0]], markersize=1)
                    # label_str = ','.join(most_common_labels)
                    label_str = most_common_labels[0]
                    # plt.text(x, y + 0.3, label_str, fontsize=1)
                else:
                    most_common_label_str, _ = Counter(label_list).most_common(1)[0]
                    plt.plot(x, y, 'o', color=colours[most_common_label_str], markersize=1)
                    # label_str = ','.join(lbl_keys)
                    label_str = ','.join(lbl_keys)
                    # plt.text(x, y + 0.3, label_str, fontsize=1)

            else:
                plt.plot(x, y, 'o', color='#DCDCDC', markersize=1)  # grey colour for untapped neurons

        plt.savefig(output_filename + '.jpeg', dpi=1200)
        plt.clf()
        plt.close('all')

    def display_interactive_gsom_nodemap(self, labels, figure_id, title, root_folder):

        fig = plt.figure(figure_id)
        ax = fig.add_subplot(111)
        plt.title(title)

        max_count = max([node.get_hit_count() for _, node in self.gsom_node_map.items()])
        listed_color_map = Display._get_color_map(max_count, alpha=0.9)

        for key, value in self.gsom_node_map.items():

            key_split = key.split(':')
            x = int(key_split[0])
            y = int(key_split[1])

            if value.get_hit_count() > 0:
                ax.plot(x, y, 'o', color=listed_color_map.colors[value.get_hit_count()], markersize=3)
                ax.text(x, y + 0.1, str(value.get_hit_count()), fontsize=4)
                # label_str = ','.join([str(labels[lbl_id]) for lbl_id in value.get_mapped_labels()])
                # ax.text(x, y + 0.3, label_str, fontsize=3)
            else:
                ax.plot(x, y, 'o', color=listed_color_map.colors[value.get_hit_count()], markersize=3)

        root_folder = root_folder

        def onclick(event, args_list):

            def get_cord(cord):
                if cord < 0:
                    return 'm' + str(cord*-1)
                else:
                    return str(cord)

            ix, iy = event.xdata, event.ydata
            x, y = int(round(ix)), int(round(iy))
            print("Clicked at x={0:5.2f}, y={1:5.2f}".format(ix, iy), "Select to show", x, y)

            image_path = args_list[0] + get_cord(x) + '_' + get_cord(y) + '.jpg'

            if os.path.exists(image_path):
                fig2 = plt.figure()
                img = mpimg.imread(image_path)
                line2 = plt.imshow(img)
                fig2.show()
            else:
                print('No cluster w.r.t. selected node')

        cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, [root_folder]))
        plt.show()

    def setup_hitcount_for_gsom_nodemap(self, figure_id, title, output_filename):

        plt.figure(figure_id)
        plt.title(title)

        max_count = max([node.get_hit_count() for _, node in self.gsom_node_map.items()])
        listed_color_map = Display._get_color_map(max_count, alpha=0.9)

        for key, value in self.gsom_node_map.items():

            key_split = key.split(':')
            x = int(key_split[0])
            y = int(key_split[1])

            if value.get_hit_count() > 0:
                plt.plot(x, y, 'o', color=listed_color_map.colors[value.get_hit_count()], markersize=2)
                plt.text(x, y + 0.1, str(value.get_hit_count()), fontsize=4)
            else:
                plt.plot(x, y, 'o', color=listed_color_map.colors[value.get_hit_count()], markersize=2)

        plt.savefig(output_filename + '.jpeg', dpi=1200)

    def display(self):
        plt.show()

    @staticmethod
    def _get_color_map(max_count, alpha=0.5):

        np.random.seed(1)

        cmap = cm.get_cmap('Reds', max_count + 1)  # set how many colors you want in color map
        # https://matplotlib.org/examples/color/colormaps_reference.html

        color_list = []
        for ind in range(cmap.N):
            c = []
            for x in cmap(ind)[:3]: c.append(x * alpha)
            color_list.append(tuple(c))

        return colors.ListedColormap(color_list, name='gsom_color_list')

    @staticmethod
    def _get_color_map_learning(max_count):
        np.random.seed(1)

        cmap = cm.get_cmap('Set1', max_count + 1)  # set how many colors you want in color map
        # https://matplotlib.org/examples/color/colormaps_reference.html

        color_tuples = cmap.colors

        colours = {}
        patches = []
        for key, col in enumerate(color_tuples):
            colours[str(key)] = col
            patches.append(mpatches.Patch(color=col, label=str(key)))

        return colours, patches

    @staticmethod
    def _get_color_map_adl_activity():
        """
        20 distinct colours
        https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        """

        # colours = {'bt': '#e6194b',
        #            'cs': '#3cb44b',
        #            'ch': '#ffe119',
        #            'ds': '#0082c8',
        #            'dg': '#f58231',
        #            'em': '#911eb4',
        #            'es': '#d2f53c',
        #            'gb': '#fabebe',
        #            'lb': '#008080',
        #            'pw': '#e6beff',
        #            'sic': '#aa6e28',
        #            'stc': '#800000',
        #            'ut': '#aaffc3',
        #            'wk': '#808000'}
        #
        # a = mpatches.Patch(color='#e6194b', label='Brush Teeth')
        # b = mpatches.Patch(color='#3cb44b', label='Climb Stairs')
        # c = mpatches.Patch(color='#ffe119', label='Comb Hair')
        # d = mpatches.Patch(color='#0082c8', label='Descend Stairs')
        # e = mpatches.Patch(color='#f58231', label='Drink Glass')
        # f = mpatches.Patch(color='#911eb4', label='Eat Meat')
        # g = mpatches.Patch(color='#d2f53c', label='Eat Soup')
        # h = mpatches.Patch(color='#fabebe', label='Getup Bed')
        # i = mpatches.Patch(color='#008080', label='Lie-down Bed')
        # j = mpatches.Patch(color='#e6beff', label='Pour Water')
        # k = mpatches.Patch(color='#aa6e28', label='Sit-down Chair')
        # l = mpatches.Patch(color='#800000', label='Stand-up Chair')
        # m = mpatches.Patch(color='#aaffc3', label='Use Telephone')
        # n = mpatches.Patch(color='#808000', label='Walk')

        colours = {'bt': '#0082c8',
                   'cs': '#3cb44b',
                   'em': '#e6194b',
                   'es': '#008080',
                   'lb': '#aa6e28'}

        b = mpatches.Patch(color='#0082c8', label='Brush Teeth')
        g = mpatches.Patch(color='#3cb44b', label='Climb Stairs')
        r = mpatches.Patch(color='#e6194b', label='Eat Meat')
        c = mpatches.Patch(color='#008080', label='Eat Soup')
        m = mpatches.Patch(color='#aa6e28', label='Liedown Bed')

        return colours, [b, g, r, c, m]

    @staticmethod
    def _get_color_map_ped_behaviour():

        colours = {'Monday': '#e6194b',
                   'Tuesday': '#3cb44b',
                   'Wednesday': '#ffe119',
                   'Thursday': '#0082c8',
                   'Friday': '#f58231',
                   'Saturday': '#911eb4',
                   'Sunday': '#d2f53c'}

        a = mpatches.Patch(color='#e6194b', label='Monday')
        b = mpatches.Patch(color='#3cb44b', label='Tuesday')
        c = mpatches.Patch(color='#ffe119', label='Wednesday')
        d = mpatches.Patch(color='#0082c8', label='Thursday')
        e = mpatches.Patch(color='#f58231', label='Friday')
        f = mpatches.Patch(color='#911eb4', label='Saturday')
        g = mpatches.Patch(color='#d2f53c', label='Sunday')

        return colours, [a, b, c, d, e, f, g]

    def _get_adl_activity_list(self):

        activities = {'bt': 'Brush Teeth',
                   'cs': 'Climb Stairs',
                   'em': 'Eat Meat',
                   'es': 'Eat Soup',
                   'lb': 'Liedown Bed'}

        return activities

    @staticmethod
    def _get_color_map_nba_basketball():

        colours = {'Moment': '#e6194b'}

        a = mpatches.Patch(color='#e6194b', label='Moment')

        return colours, [a]
