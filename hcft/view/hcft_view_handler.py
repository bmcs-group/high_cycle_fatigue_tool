import traitsui.api as ui
import traits.api as tr

from hcft.about_tool import AboutTool
from hcft.utils.csv_joiner import CSVJoiner

menu_exit = ui.Action(name='Exit', action='menu_exit')

menu_utilities_csv_joiner = ui.Action(name='CSV Joiner', action='menu_utilities_csv_joiner')

menu_about_tool = ui.Action(name='About', action='menu_about_tool')

class ViewHandler(ui.Handler):

    # You can initialize this by obtaining it from the methods below and, self.info = info
    info = tr.Instance(ui.UIInfo)

    def menu_utilities_csv_joiner(self):
        csv_joiner = CSVJoiner()
        # kind='modal' pauses the background traits window until this window is closed
        csv_joiner.configure_traits()

    def menu_about_tool(self):
        about_tool = AboutTool()
        about_tool.configure_traits()

    def menu_exit(self, info):
        if info.initialized:
            info.ui.dispose()
