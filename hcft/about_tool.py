'''
Created on 15 Apr 2020

@author: Homam
'''
import traits.api as tr
import traitsui.api as ui

from hcft.app_icon import app_icon
from hcft.version import __version__


class AboutTool(tr.HasStrictTraits):
    about_tool_text = tr.Str(
        'High-Cycle Fatigue Tool \nVersion: ' + __version__ + '\n\nHCFT is a tool with a graphical user interface \nfor processing CSV files obtained from fatigue \nexperiments up to the high-cycle fatigue ranges.\nAdditionally, tests with monotonic loading can be processed.\n\nDeveloped in:\nRWTH Aachen University - Institute of Structural Concrete\nBy:\nDr.-Ing. Homam Spartali\nProf. Dr. habil. Rostislav Chudoba\n\nGithub link:\nhttps://github.com/bmcs-group/high_cycle_fatigue_tool')

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
        icon=app_icon,
        resizable=True,
        width=0.3,
        height=0.25
    )


if __name__ == '__main__':
    about_tool = AboutTool()
    about_tool.configure_traits()
