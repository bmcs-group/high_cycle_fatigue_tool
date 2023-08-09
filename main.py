
import os
from hcft.api import HCFT
from bmcs_utils.api import set_latex_mpl_format

# To use Sans font in matplotlib
set_latex_mpl_format(font_size=18)

hcft = HCFT()
hcft.configure_traits()
