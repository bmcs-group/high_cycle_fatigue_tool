from pathlib import Path

from pyface.image_resource import ImageResource

RESOURCES_DIR = Path(__file__).parent / 'resources'
ICON_FILE = RESOURCES_DIR / 'hcft_icon.png'

# TraitsUI replaces the application icon of every window by its default icon, unless the view defines its own icon,
# therefore, this must be passed as icon to all views that open a window
app_icon = ImageResource(ICON_FILE.name, search_path=[str(RESOURCES_DIR)])
