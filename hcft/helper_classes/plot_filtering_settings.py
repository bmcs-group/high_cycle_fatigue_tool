import traits.api as tr


class PlotSettings(tr.HasStrictTraits):
    num_of_first_rows_to_take = tr.Range(low=0, high=10 ** 9, value=6000, mode='spinner')
    num_of_rows_to_skip_after_each_section = tr.Range(low=0, high=10 ** 9, value=20000, mode='spinner')
    num_of_rows_in_each_section = tr.Range(
        low=0, high=10 ** 9, value=200, mode='spinner')
