import traits.api as tr
import traitsui.api as ui
from traitsui.extras.checkbox_column import CheckboxColumn

average_columns_editor = ui.TableEditor(
    sortable=False,
    configurable=False,
    auto_size=False,
    columns=[CheckboxColumn(name='selected', label='Select', width=0.12),
             ui.ObjectColumn(name='column_name', editable=False, width=0.24,
                             horizontal_alignment='left')])


class Column(tr.HasStrictTraits):
    column_name = tr.Str
    selected = tr.Bool(False)


class ColumnsAverage(tr.HasStrictTraits):
    columns = tr.List(Column)

    # Trait view definitions:
    traits_view = ui.View(
        ui.Item('columns',
                show_label=False,
                editor=average_columns_editor
                ),
        buttons=[ui.OKButton, ui.CancelButton],
        title='Select data columns to be averaged',
        width=0.15,
        height=0.3,
        resizable=True
    )
