import re
import textwrap
import uuid
import dateutil.parser as duparser

import networkx as nx
from pyvis.network import Network

from elephant.buffalo.types import ObjectInfo, VarArgs
import datetime

# TODO: get rid of pyvis in favor of prov
now = datetime.datetime.now()


def get_gephi_interval(analysis_step):
    return duparser.parse(analysis_step.time_stamp_start).timestamp(),\
           now.timestamp()


class ObjectDescription(object):

    attributes = None
    annotations = None
    array_annotations = None
    converters = {}

    def _add_line(self, key, value, html=True):
        converter = self.converters.get(key)
        value = converter(value) if converter is not None else value
        if html:
            return f"<br><font size=\"0.5\"><i>{key}</i>={value}</font>"
        return str(value)

    def build_title(self, object_details, array_summary=True):
        title = ""
        attrs = {}

        # Get values of the selected attributes
        if self.attributes is not None:
            attributes = list(object_details.keys()) if \
                self.attributes == 'all' else self.attributes
            for attr in attributes:
                if attr in object_details:
                    title += self._add_line(attr, object_details[attr])
                    attrs[attr] = self._add_line(attr, object_details[attr], html=False)

        # Get values of the selected annotations
        if self.annotations is not None and 'annotations' in object_details:
            annotations_dict = object_details['annotations']
            for attr in self.annotations:
                if attr in annotations_dict:
                    title += self._add_line(attr, annotations_dict[attr])
                    attrs[attr] = self._add_line(attr, annotations_dict[attr], html=False)

        if (self.array_annotations is not None and
                'array_annotations' in object_details):
            annotations_dict = object_details['array_annotations']
            for attr in self.array_annotations:
                if attr in annotations_dict:
                    value = annotations_dict[attr]
                    if array_summary:
                        value = set(value)
                    title += self._add_line(attr, value)
                    attrs[attr] = self._add_line(attr, value, html=False)

        return title, attrs

    def __init__(self, attributes=None, annotations=None,
                 array_annotations=None, converters=None):
        # Update static definitions if additional configurations are passed
        if attributes is not None:
            if self.attributes is None or self.attributes == 'all':
                self.attributes = attributes
            else:
                self.attributes.extend(attributes)

        if annotations is not None:
            if self.annotations is None:
                self.annotations = annotations
            else:
                self.annotations.extend(annotations)

        if array_annotations is not None:
            if self.array_annotations is None:
                self.array_annotations = array_annotations
            else:
                self.array_annotations.extend(array_annotations)

        if converters is not None:
            self.converters.update(converters)


class FunctionParameters(ObjectDescription):
    attributes = 'all'


class ArrayDescription(ObjectDescription):
    attributes = ['shape', 'dtype']


def _convert_units(value):
    return value.dimensionality


def _convert_channel_names(value):
    value = sorted(value,
                   key=lambda x: int([c for c in re.split(r'(\d+)', x)][1]))
    tokens = []
    strings = []
    token_count = 0
    for value in value:
        token_count += len(value) + 1
        if token_count < 60:
            tokens.append(value)
        else:
            strings.append(",".join(tokens))
            tokens.clear()
            tokens.append(value)
            token_count = len(value) + 1

    strings.append(",".join(tokens))
    return ",<br>".join(strings)


def _convert_name(value):
    if value is None:
        return str(None)
    return textwrap.shorten(value, 30)


class NeoDescription(ObjectDescription):
    attributes = ['shape', 'units', 't_start', 't_stop', 'name']
    annotations = ['trial_number', 'implantation_site', 'nix_name',
                   'trial_protocol', 'recording_area']
    array_annotations = ['channel_names', 'belongs_to_trialtype',
                         'trial_event_labels', 'trial_number',
                         'implantation_site']
    converters = {'units': _convert_units,
                  'channel_names': _convert_channel_names,
                  'name': _convert_name}


def _convert_params(value):
    return "<br>".join(textwrap.wrap(f"{value}", 60))


class AnalysisObjectDescription(ObjectDescription):
    attributes = ['pid', 'create_time', 'name', 'params', 'method']
    converters = {'params': _convert_params}


class MatplotlibDescription(ObjectDescription):
    attributes = ['title']


DEFAULT_DESCRIPTION_MAP = {
    'neo.core.event.Event': NeoDescription(),
    'neo.core.block.Block': NeoDescription(),
    'neo.core.segment.Segment': NeoDescription(),
    'neo.core.epoch.Epoch': NeoDescription(),
    'neo.core.analogsignal.AnalogSignal': NeoDescription(),
    'neo.core.spiketrain.SpikeTrain': NeoDescription(),
    'quantities.quantity.Quantity': NeoDescription(),
    'numpy.ndarray': ArrayDescription(),
    'elephant.buffalo.objects.spectral.PSDObject': AnalysisObjectDescription(),
    '_function': FunctionParameters()
}


class BuffaloProvenanceGraph(nx.DiGraph):
    """
    Directed Acyclic Graph representing the provenance history.
    This class is inherited from the Networkx `DiGraph` object.

    Parameters
    ----------
    history : list of AnalysisStep
        The history of steps tracked in the Python script, that will be
        represented in the graph.
    description_map : dict, optional
        Dictionary containing the mapping of each object type (as `str`) to
        an `ObjectDescription` instance. This is used to format and display
        the information on the graph as labels.
        If None, the default description map will be used.
        Default: None
    """

    def __init__(self, history, description_map=None):
        super(BuffaloProvenanceGraph, self).__init__()
        if description_map is None:
            description_map = DEFAULT_DESCRIPTION_MAP
        self.description_map = description_map
        for step in history:
            self.add_step(step)

    def _add_input_to_output(self, analysis_step, input_obj, edge_label,
                             edge_title, multi_output, function_edge, **attrs):

        def _connect_edge(input_obj, output_obj):
            start, end = get_gephi_interval(analysis_step)

            if function_edge:
                title, objattrs = self.description_map['_function'].build_title(
                    analysis_step.params)
                x, y = self._get_x_y(analysis_step.vis, 0, function_edge)
                self.add_node(function_edge, label=edge_label,
                              title=edge_title+title, type='function',
                              params=analysis_step.params,
                              x=x, y=y, start=start, end=end)
                if input_obj is not None:
                    self.add_edge(input_obj.hash, function_edge, type='input',
                                  start=start, end=end, **attrs)
                if output_obj is not None:
                    self.add_edge(function_edge, output_obj.hash, type='output',
                                  start=start, end=end, **attrs)
            else:
                self.add_edge(input_obj.hash, output_obj.hash, label=edge_label,
                              title=edge_title, params=analysis_step.params,
                              type='static', start=start, end=end, **attrs)

        start, end = get_gephi_interval(analysis_step)

        if input_obj is not None:
            obj_type, obj_label, node_type, objattrs = self._get_type_and_label(input_obj)
            x, y = self._get_x_y(analysis_step.vis, -1, input_obj.hash)

            self.add_node(input_obj.hash, label=obj_label, title=obj_type,
                          type=node_type, x=x, y=y, start=start, end=end, **objattrs)

        if multi_output:
            for output_key, output_obj in analysis_step.output.items():
                _connect_edge(input_obj, output_obj)
        else:
            if analysis_step.output:
                index = next(iter(analysis_step.output))
                output_obj = analysis_step.output[index]
            else:
                output_obj = None
            _connect_edge(input_obj, output_obj)

    def _get_type_and_label(self, obj):
        attrs = {}
        if isinstance(obj, ObjectInfo):
            # Provenance of a Python object (ObjectInfo named tuple)
            obj_type = obj.type
            obj_label = obj_type.split(".")[-1]
            title = ""
            if obj_type in self.description_map:
                title, attrs = self.description_map[obj_type].build_title(obj.details)
            return obj_type + title, obj_label, "data", attrs

        # Provenance of a file (FileInfo named tuple)
        return f"{obj.path}<br>[{obj.hash_type}:{obj.hash}]", "<File>", "file", attrs

    @staticmethod
    def _get_edge_attrs_and_labels(analysis_step):
        edge_attr = analysis_step.params
        edge_label = analysis_step.function.name
        edge_title = edge_label

        function_edge = None
        if edge_label == 'attribute':
            edge_label = f".{edge_attr['name']}"
        elif edge_label == 'subscript':
            if 'slice' in edge_attr:
                subscript_information = str(edge_attr['slice'])
            else:
                subscript_information = edge_attr['index']
            edge_label = f"[{subscript_information}]"
        else:
            if analysis_step.function.module:
                edge_title = f"{analysis_step.function.module}.{edge_title}"
            edge_label = edge_title.split(".")[-1]
            function_edge = hash(str(uuid.uuid4()))

        return edge_label, edge_title, function_edge

    def _get_x_y(self, vis, factor, obj):
        if obj in self.nodes:
            if self.nodes[obj]['x'] is not None:
                return self.nodes[obj]['x'], self.nodes[obj]['y']
        if vis[0] is not None:
            return vis[0], vis[1]*2+factor
        return -1, -1

    def add_step(self, analysis_step, **attr):

        start, end = get_gephi_interval(analysis_step)

        for key, obj in analysis_step.output.items():
            obj_type, obj_label, node_type, objattrs = self._get_type_and_label(obj)
            x, y = self._get_x_y(analysis_step.vis, 1, obj.hash)
            self.add_node(obj.hash, label=obj_label, title=obj_type,
                          type=node_type, x=x, y=y, start=start, end=end, **objattrs)
        multi_output = len(analysis_step.output.keys()) > 1

        edge_label, edge_title, function_edge = \
            self._get_edge_attrs_and_labels(analysis_step)

        if analysis_step.input.keys():
            for key, obj in analysis_step.input.items():
                if isinstance(obj, VarArgs):
                    for var_arg in obj.args:
                        self._add_input_to_output(analysis_step, var_arg,
                                                  edge_label, edge_title,
                                                  multi_output, function_edge,
                                                  **attr)
                else:
                    self._add_input_to_output(analysis_step, obj, edge_label,
                                              edge_title, multi_output,
                                              function_edge, **attr)
        else:
            # Function without input
            self._add_input_to_output(analysis_step, None, edge_label,
                                      edge_title, multi_output,
                                      function_edge, **attr)

    def _adjust_node_positions(self):
        nodes = nx.algorithms.topological_sort(self)
        for idx, node in enumerate(nodes):
            self.nodes[node]['y'] = idx

    def to_pyvis(self, filename, show=False, grayscale=True):
        """
        This method takes an existing Networkx graph and translates
        it to a PyVis graph format that can be accepted by the VisJS
        API in the Jinja2 template.

        Parameters
        ----------
        filename : str
            Destination where to save the file. Must be an HTML file.
        show : bool, optional
            If True, display the graph in the browser after saving.
            Default: False.
        grayscale : bool, optional
            If True, use grayscale colors for all nodes, that will change
            color when selected.
            If False, all nodes will be colored according to the node type.
            Default: True.
        """

        def _get_color(highlight):
            # Return the string or dictionary with the proper node
            # colors according to the visualization options
            if not grayscale:
                return highlight

            return {
                        "background": "rgba(120, 120, 120, 1)",
                        "border": "rgba(120, 120, 120, 1)",
                        "highlight": {
                            "background": highlight,
                            "border": highlight,
                        }
                    }

        def _add_node(node_id):
            # Add the given node to the pyvis Network `net`.
            attr = nodes[node_id]

            # Get the node predefined X/Y coordinates and scale them
            x_pos = attr.get('x')
            y_pos = attr.get('y')
            position = {}
            if x_pos is not None:
                position = {'x': x_pos * 500, 'y': y_pos * 100}

            # Set shape, color and font attributes of the node
            node_type = attr['type']
            shape = shape_types[node_type]
            color = _get_color(color_types[node_type])
            font = {"color": "rgba(255, 255, 255, 1)"}

            net.add_node(hash(node_id), shape=shape, color=color,
                         label=attr['label'], title=attr['title'],
                         physics=False, font=font, **position)

        # Set shape associated with each node type
        # - data     = input/output from functions
        # - function = function tracked during the execution
        # - file     = file read or written
        shape_types = {'data': 'ellipse',
                       'function': 'box',
                       'file': 'database'}

        # Set color associated with each node type
        color_types = {'data': 'blue',
                       'function': 'red',
                       'file': 'green'}

        #self._adjust_node_positions()

        edges = self.edges.data()
        nodes = self.nodes

        # Initializes the pyvis network graph
        net = Network(height="960px", width="1280px", directed=True)

        # Loop over all edges from the Networkx graph, adding the in/out
        # nodes and setting the edges
        for v, u, edge_attr in edges:
            _add_node(v)
            _add_node(u)
            labels = {key: edge_attr[key] for key in ('label', 'title')
                      if key in edge_attr}
            net.add_edge(hash(v), hash(u), **labels)

        # Set pyvis options
        net.options.edges.arrowStrikethrough = False
        net.options.physics.enabled = False
        net.options.interaction.multiselect = True
        net.options.interaction.tooltipDelay = 0
        net.options.navigationButons = True

        # Display control buttons on the HTML file
        net.show_buttons()

        # Save the graph
        net.save_graph(filename)

        # Open the file for visualization
        if show:
            net.show(name=filename)

    def save_gexf(self, path):
        nx.write_gexf(self, path)

