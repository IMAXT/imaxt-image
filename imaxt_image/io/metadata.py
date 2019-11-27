import xml.etree.ElementTree as ET
from typing import Dict

import xmltodict


class Metadata:
    """Metadata reader.

    Parameters
    ----------
    description
        XML document

    """

    def __init__(self, description: str):
        self._description = description
        self.tree = ET.fromstring(self._description)

    def as_dict(self) -> Dict:
        """Return a dictionary representation of the XML metadata.
        """
        return xmltodict.parse(self._description)

    @property
    def description(self) -> str:
        return self._description
