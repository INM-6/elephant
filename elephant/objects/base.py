"""
This module implements a base class for standardized analysis output objects
in Elephant.

:copyright: Copyright 2014-2022 by the Elephant team, see `doc/authors.rst`.
:license: BSD, see LICENSE.txt for details.
"""

from datetime import datetime
from copy import deepcopy
import uuid


class AnalysisObject(object):
    """
    Base class for analysis output in Elephant.

    This class supports storage of basic provenance information.
    """

    def __init__(self, *args, **kwargs):
        self._create_time = datetime.now()
        self._warnings_raised = False
        self._ontology_path = None
        self._provenance_track = None

        # If base class does not have annotations, we store them in this class
        if not hasattr(self, 'annotations'):
            self.annotations = dict()

        self._pid = self._create_pid()

    def _store_kwargs(self, kwargs):
        # Stores additional parameters as additional object attributes.
        # This can be used internally by child classes to provide flexibility
        # when initializing the object with extended provenance information.
        for key, value in kwargs.items():

            setattr(self, key, value if not isinstance(value, (list, dict))
                    else deepcopy(value))

    @staticmethod
    def _create_pid():
        # TODO: create globally unique ID
        return uuid.uuid1()

    @property
    def warnings_raised(self):
        """
        If any warning was raised during the creation of the object

        Returns
        -------
        bool
            True if warning raised, False otherwise.
        """
        return self._warnings_raised

    @property
    def create_time(self):
        """
        Time of object creation.

        Returns
        -------
        datetime.datetime
        """
        return self._create_time

    @property
    def pid(self):
        """
        Persistent ID associated with the object.

        Returns
        -------
        str
        """
        return self._pid

    @property
    def ontology_path(self):
        return self._ontology_path

    def annotate(self, key, value):
        """
        Insert annotation in the object.

        Parameters
        ----------
        key : hashable type
            Key in the annotation dictionary.
        value : object
            Value associated with the key.
        """
        self.annotations[key] = value

    def set_annotations(self, annotations):
        """
        Inserts a dictionary into the object annotations.

        Parameters
        ----------
        annotation : dict
            Dictionary with object annotations.
        """
        if not isinstance(annotations, dict):
            raise TypeError("Annotations must be a dictionary")
        self.annotations.update(annotations)

    def serialize(self, path):
        """
        Save object to disk.

        Parameters
        ----------
        path : str or Path
            Destination of object.
        """
        raise NotImplementedError

    def load(self, path):
        """
        Loads object from a file.

        Parameters
        ----------
        path : str or Path
            Source of the object on disk.
        """
        raise NotImplementedError
