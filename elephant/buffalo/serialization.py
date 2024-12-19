"""
This module implements functionality to serialize the provenance track using
the W3C Provenance Data Model (PROV).
"""

import pathlib
from itertools import product

from prov.model import ProvDocument, PROV, PROV_TYPE, Namespace

from elephant.buffalo.object_hash import BuffaloFileHash
from elephant.buffalo.types import ObjectInfo, FileInfo, VarArgs

import quantities as pq


# Values that are used to compose the URNs
# URNs take the general form "urn:NID:NSS", followed by optional components
# according to RFC 8141.

NID_ALPACA = "alpaca"

NSS_FUNCTION = "function"             # Functions executed
NSS_FILE = "file"                     # Files accessed
NSS_DATA = "object"                   # Data objects (input/outputs/containers)
NSS_SCRIPT = "script"                 # The execution script
NSS_PARAMETER = "parameter"           # Function parameter
NSS_ATTRIBUTE = "attribute"           # Data object attribute (i.e., class)
NSS_ANNOTATION = "annotation"         # Data object annotation (e.g., Neo)
NSS_CONTAINER = "container"           # For storing container membership access information


ALL_NSS = [NSS_FUNCTION, NSS_FILE, NSS_DATA, NSS_SCRIPT, NSS_PARAMETER,
           NSS_ATTRIBUTE, NSS_ANNOTATION, NSS_CONTAINER]


# Other namespaces used

RDFS = Namespace("rdfs", "http://www.w3.org/2000/01/rdf-schema#")


# Conversion functions, that will ensure that attributes and metadata are
# represented as valid strings according to XSD literals and other classes
# of the PROV model (e.g., containers and members)

def _quantity_to_prov(value):
    return str(value)


def _list_to_prov(value):
    return str(value)


def _neo_to_prov(value):
    # For Neo objects, we create a lightweight representation, to avoid
    # dumping all the information such as SpikeTrain timestamps and Event
    # times. They will be represented as additional entities
    type_information = type(value)
    neo_class = f"{type_information.__module__}.{type_information.__name__}"
    repr = f"{neo_class}("
    counter = 0
    for attribute in ('t_start', 't_stop', 'shape', 'dtype', 'name',
                      'description', 'nix_name'):
        if hasattr(value, attribute):
            if counter > 0:
                repr += ", "
            attr_value = getattr(value, attribute)
            repr += f"{attribute}={_ensure_type(attr_value)}"
            counter += 1
    repr += ")"
    return repr


_converters_map = {
    pq.Quantity: _quantity_to_prov,
    list: _list_to_prov,
}


def _ensure_type(value):
    value_type = type(value)
    package = str(value_type).split(".")[0]

    if package == 'neo':
        return _neo_to_prov(value)
    if value_type in _converters_map:
        return _converters_map[value_type](value)
    if isinstance(value, (int, str, bool, float)):
        return value
    return str(value)


# Functions to generate URIs and attribute dictionaries used with the
# activities and entities

def _get_entity_uri(object_info):
    if isinstance(object_info, ObjectInfo):
        return f"{NSS_DATA}:{object_info.type}:{object_info.hash}"
    elif isinstance(object_info, FileInfo):
        return f"{NSS_FILE}:{object_info.hash_type}:{object_info.hash}"
    return None


def _get_entity_attributes(object_info):
    attributes = {}
    if isinstance(object_info, ObjectInfo):
        package_name = object_info.type.split(".")[0]
        for name, value in object_info.details.items():
            # Handle Neo objects, to avoid dumping all the information
            # in collections such as `segments` or `events`
            if package_name == 'neo':
                if name in ('segments', 'events', 'analogsignals',
                            'spiketrains', 'channel_indexes', 'block',
                            'segment', 'epochs', 'parent'):
                    if isinstance(value, list):
                        # Extract the readable name of each Neo object in the
                        # list
                        attr_value = "["
                        counter = 0
                        for item in value:
                            if counter > 0:
                                attr_value += ", "
                            attr_value += _neo_to_prov(item)
                            counter += 1
                        attr_value += "]"
                    else:
                        # Just get the readable name of the Neo object
                        attr_value = _neo_to_prov(value)
                    attr_key = f"{NSS_ATTRIBUTE}:{name}"
                    attributes[attr_key] = attr_value

                elif name == 'annotations' and isinstance(value, dict):
                    # Handle annotations in Neo objects
                    for annotation, annotation_value in value.items():

                        # Make sure that types such as list and Quantity are
                        # handled
                        annotation_value = _ensure_type(annotation_value)

                        attr_key = f"{NSS_ANNOTATION}:{annotation}"
                        attributes[attr_key] = annotation_value
                else:
                    # Other attributes
                    value = _ensure_type(value)
                    attr_key = f"{NSS_ATTRIBUTE}:{name}"
                    attributes[attr_key] = value

            else:
                # Make sure that types such as list and Quantity are handled
                value = _ensure_type(value)
                attr_key = f"{NSS_ATTRIBUTE}:{name}"
                attributes[attr_key] = value

    return attributes


def _get_activity_uri(function_info):
    activity_uri = f"{NSS_FUNCTION}:"
    if function_info.module:
        activity_uri = f"{activity_uri}{function_info.module}."
    activity_uri = f"{activity_uri}{function_info.name}"
    return activity_uri


def _get_activity_params(function_params):
    params = {}
    for name, value in function_params.items():
        value = _ensure_type(value)
        param_key = f"{NSS_PARAMETER}:{name}"
        params[param_key] = value
    return params


def _get_membership_params(function_params):
    params = {}
    for name, value in function_params.items():
        if name == "index":
            param_key = f"{NSS_CONTAINER}:index"
        elif name == "slice":
            param_key = f"{NSS_CONTAINER}:slice"
        elif name == "name":
            param_key = f"{NSS_CONTAINER}:attribute"
        params[param_key] = value
    return params


class BuffaloProvDocument(ProvDocument):

    def __init__(self, script_file_name):
        super().__init__(records=None, namespaces=None)

        # Set default namespace to avoid repetitions of the prefix
        for namespace in ALL_NSS:
            self.add_namespace(Namespace(namespace,
                                         f"urn:{NID_ALPACA}:{namespace}:"))

        # Add other namespaces
        self.add_namespace(RDFS)

        # Create an Agent to represent the script
        # We are using the SoftwareAgent subclass for the description
        # URN defined as :script:script_name:file_hash
        script_hash = BuffaloFileHash(script_file_name).info().hash
        script_name = pathlib.Path(script_file_name).name
        script_uri = f"{NSS_SCRIPT}:{script_name}?{script_hash}"
        script_attributes = {PROV_TYPE: PROV["SoftwareAgent"],
                             "rdfs:label": script_file_name}
        self._script_agent = self.agent(script_uri, script_attributes)

    def _create_entity(self, info):
        # Create a PROV Entity based on ObjectInfo/FileInfo information

        if isinstance(info, VarArgs):
            # If this is a VarArgs, several objects are inside.
            # Process recursively
            for var_arg in info:
                self._create_entity(var_arg)
        else:
            uri = _get_entity_uri(info)
            if uri:
                metadata = _get_entity_attributes(info)
                try:
                    return self.entity(identifier=uri, other_attributes=metadata)
                except Exception as e:
                    print(e)

    def _add_analysis_step(self, step):

        def _is_membership(function):
            name = function.name
            return name in ("attribute", "subscript")

        # Add one `AnalysisStep` record and generate all the provenance
        # semantic relationships

        function = step.function
        if _is_membership(function):
            # attributes and subscripting operations
            activity_params = _get_membership_params(step.params)
            container = step.input[0]
            child = step.output[0]
            container_entity = self._create_entity(container)
            child_entity = self._create_entity(child)
            child_entity.add_attributes(activity_params)
            container_entity.hadMember(child_entity)
        else:
            # This is a function execution
            activity_uri = _get_activity_uri(function)
            activity_params = _get_activity_params(step.params)
            cur_activity = self.activity(identifier=activity_uri)

            step_time = step.time_stamp_end

            # Add all the inputs as entities, and create a `used` association with
            # the activity. URNs differ when the input is a file or Python object.
            # This is a qualified relationship, as the attributes are stored
            # together with the timestamp
            input_entities = []
            for key, value in step.input.items():
                cur_entity = self._create_entity(value)
                input_entities.append(cur_entity)
                self.used(activity=cur_activity,
                          entity=cur_entity,
                          time=step_time,
                          other_attributes=activity_params)

            # Add all the outputs as entities, and create the `wasGenerated`
            # relationship. This is a qualified relationship, as the attributes
            # will be stored together with the timestamp.
            output_entities = []
            for key, value in step.output.items():
                cur_entity = self._create_entity(value)
                output_entities.append(cur_entity)
                self.wasGeneratedBy(entity=cur_entity,
                                    activity=cur_activity,
                                    time=step_time,
                                    other_attributes=activity_params)

            # Iterate over the input/output pairs to add the `wasDerived`
            # relationship
            for input_entity, output_entity in \
                    product(input_entities, output_entities):
                self.wasDerivedFrom(generatedEntity=output_entity,
                                    usedEntity=input_entity)

            # Attribute the activity to the script
            self.wasAttributedTo(entity=cur_activity, agent=self._script_agent)

    @classmethod
    def read_records(cls, file_name, file_format='rdf'):
        """
        Reads PROV data that was previously serialized.

        Parameters
        ----------
        file_name : str or path-like
            Location of the file with PROV data to be read.
        file_format : {'json', 'rdf', 'ttl', 'provn', 'xml'}
            Format used in the file that is being read.
            Turtle files (*.ttl) are treated as RDF files.
            If None, the format will be inferred from the extension.
            Default: None

        Returns
        -------
        BuffaloProvDocument
            Instance with the loaded provenance data, that can be further
            used for plotting/querying.

        Raises
        ------
        ValueError
            If `file_format` is None and `file_name` has no extension to infer
            the format.
            If `file_format` is not 'rdf', 'ttl', 'json', 'prov', or 'xml'.
        """
        file_location = pathlib.Path(file_name)

        if file_format is None:
            extension = file_location.suffix
            if not extension.startswith('.'):
                raise ValueError("File has no extension and no format was "
                                 "provided")
            file_format = extension[1:]

        if file_format == 'ttl':
            file_format = 'rdf'
        elif file_format not in ['rdf', 'json', 'provn', 'xml']:
            raise ValueError("Unsupported serialization format")

        with open(file_name, "r") as source:
            return cls.deserialize(source, format=file_format)

    def process_analysis_steps(self, analysis_steps):
        for step in analysis_steps:
            self._add_analysis_step(step)
