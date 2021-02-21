'''
Created on 15 Apr 2020

@author: Homam
'''
import traits.api as tr
import traitsui.api as ui
import pyface.api as pf
import numpy as np
import pandas as pd
import os
from itertools import takewhile, repeat

from pyface.message_dialog import MessageDialog
from traitsui.editors import ProgressEditor


class AboutTool(tr.HasStrictTraits):
    about_tool_text = tr.Str(
        'High-Cycle Fatigue Tool \nVersion: 1.0.0\n\nHCFT is a tool with a graphical user interface \nfor processing CSV files obtained from fatigue \nexperiments up to the high-cycle fatigue ranges.\nAdditionally, tests with monotonic loading can be processed.\n\nDeveloped in:\nRWTH Aachen University - Institute of Structural Concrete\nBy:\nDr.-Ing. Rostislav Chudoba\nM.Sc. Homam Spartali\n\nGithub link:\nhttps://github.com/ishomam/high-cycle-fatigue-tool')

    # =========================================================================
    # Configuration of the view
    # =========================================================================
    traits_view = ui.View(
        ui.VGroup(
            ui.UItem('about_tool_text', style='readonly'),
            show_border=True
        ),
        buttons=[ui.OKButton],
        title='About HCFT',
        resizable=True,
        width=0.3,
        height=0.25
    )


if __name__ == '__main__':
    about_tool = AboutTool()
    about_tool.configure_traits()
