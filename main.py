
import os
from hcft.api import HCFT
from bmcs_utils.api import set_latex_mpl_format

# To use Sans font in matplotlib
set_latex_mpl_format()

hcft = HCFT(file_path=os.path.expanduser("~"))
hcft.configure_traits()