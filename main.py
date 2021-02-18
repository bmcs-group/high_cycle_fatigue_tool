
import os
from hcft.api import HCFT
hcft = HCFT(file_path=os.path.expanduser("~"))
hcft.configure_traits()