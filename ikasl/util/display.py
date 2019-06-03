
import functools
import heapq
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

from util import cluster_viewer

import graphviz as gv


class Display:

    def __init__(self, ikasl_structure):
        self.ikasl = ikasl_structure
        self.graph = functools.partial(gv.Digraph, format='svg')()
        self.viewer_image_clusters = {}

    def display_tree(self, graph_name, output_filename, weight_labels=None, labels_to_show=5, enable_style=False, save_format='svg'):
        self.graph = self._draw_graph(self.graph, weight_labels, labels_to_show)
        self.graph.format = save_format

        if enable_style:
            self.graph = self.apply_styles(self.graph, graph_name)

        self.graph.render(output_filename)

    def view_clusters(self, image_files_root_folder, width, height, sequence_to_show):
        if len(self.viewer_image_clusters) < 1:
            return
        viewer = cluster_viewer.Viewer(image_files_root_folder, width, height, sequence_to_show)
        viewer.view(self.viewer_image_clusters[sequence_to_show])

    def show_text(self):
        itr = 0
        for key, ikasl_layer in self.ikasl.items():
            print('Step', key)
            for node in ikasl_layer.generalisation_layer.aggregated_nodemap:
                if len(node.input_vector_weights) > 0:
                    print('\tPathway', node.get_pathway_id(), ':', end=' ')
                    for input_weight in node.input_vector_weights:
                        print(input_weight.weight_label, end=' ')
                    print('\n')
            itr += 1

    def _draw_graph(self, graph, weight_labels, labels_to_show):

        root_node_name = '0,0'
        nodes, edges = [], []
        nodes.append((root_node_name, {'label': 'Begin', 'color': 'red', 'shape': 'oval'}))

        parent_pathways = self._get_parent_pathways()

        for key, ikasl_layer in self.ikasl.items():

            self.viewer_image_clusters[key] = {}

            for node in ikasl_layer.generalisation_layer.aggregated_nodemap:

                node_pathway_id = node.get_pathway_id() + 1
                node_parent_pathway_id = node.get_parent_pathway_id() + 1

                node_id = str(key+1) + '.' + str(node_pathway_id)
                parent_node_id = str(key) + '.' + str(node_parent_pathway_id)

                node_name = 'Cluster ' + node_id
                node_text = node_name

                # Select the highest weighted labels
                if weight_labels is not None:
                    selected_labels = ''
                    label_counter = 0
                    for label_id in heapq.nlargest(labels_to_show, range(len(node.weights)), node.weights.__getitem__):
                        if label_counter > 0 and (label_counter % 5 == 0):
                            selected_labels += '\n'
                        selected_labels += weight_labels[label_id] + ' '
                        label_counter += 1
                    node_text += '\n' + selected_labels

                # Select the items categorised into the cluster
                if len(node.input_vector_weights) > 0:
                    frames = ''
                    for input_weight in node.input_vector_weights:
                        frames += str((input_weight.weight_label + 1)) + ' '
                    node_text += '\n' + frames

                    # Include frames into the image cluster database
                    self.viewer_image_clusters[key][('Pathway '+str(node_pathway_id))] = frames

                if (key < (len(self.ikasl)-1)) and (node.get_pathway_id() in parent_pathways[key]):
                    nodes.append((node_id, {'label': node_text, 'color': 'red', 'shape': 'oval'}))
                else:
                    nodes.append((node_id, {'label': node_text}))

                if key > 0:
                    edges.append(((parent_node_id, node_id), {'label': 'Pw: '+str(node_pathway_id)}))
                else:
                    edges.append(((root_node_name, node_id), {'label': 'Pw: '+str(node_pathway_id)}))

        return self.add_edges(self.add_nodes(graph, nodes), edges)

    def _get_parent_pathways(self):
        encountered_pathways = {}
        parents = {}
        for key, ikasl_layer in self.ikasl.items():
            layer_pathways = []
            layer_parents = []
            for node in ikasl_layer.generalisation_layer.aggregated_nodemap:
                layer_pathways.append(node.get_pathway_id())
                prev_key = (int(key)-1)
                if (int(key) > 0) and (node.get_pathway_id() not in encountered_pathways[prev_key]):
                    layer_parents.append(node.get_parent_pathway_id())
            encountered_pathways[key] = list(layer_pathways)
            parents[(int(key) - 1)] = list(layer_parents)
        return parents

    @staticmethod
    def apply_styles(graph, graph_name):

        """
        Mode styling guidelines
        https://stackoverflow.com/questions/13814640/color-a-particular-node-in-networkx-and-graphviz
        """
        # styles = {
        #     'graph': {
        #         'label': graph_name,
        #         'rankdir': 'LR',
        #     },
        # }

        styles = {
            'graph': {
                'label': graph_name,
                'fontsize': '16',
                'fontcolor': 'white',
                'bgcolor': '#333333',
                'rankdir': 'TB',
            },
            'nodes': {
                'fontname': 'Helvetica',
                'shape': 'box',  # box, hexagon, oval
                'fontcolor': 'white',
                'color': 'white',
                'style': 'filled',
                'fillcolor': '#006699',
            },
            'edges': {
                'style': 'dashed',
                'color': 'white',
                'arrowhead': 'open',
                'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'white',
            }
        }

        graph.graph_attr.update(
            ('graph' in styles and styles['graph']) or {}
        )
        graph.node_attr.update(
            ('nodes' in styles and styles['nodes']) or {}
        )
        graph.edge_attr.update(
            ('edges' in styles and styles['edges']) or {}
        )
        return graph

    @staticmethod
    def add_nodes(graph, nodes):
        for n in nodes:
            if isinstance(n, tuple):
                graph.node(n[0], **n[1])
            else:
                graph.node(n)
        return graph

    @staticmethod
    def add_edges(graph, edges):
        for e in edges:
            if isinstance(e[0], tuple):
                graph.edge(*e[0], **e[1])
            else:
                graph.edge(*e)
        return graph
