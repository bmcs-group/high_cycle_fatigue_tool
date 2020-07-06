import traits.api as tr


class PlotSettings(tr.HasStrictTraits):
    first_rows = tr.Range(low=0, high=10 ** 9, value=6000, mode='spinner')
    distance = tr.Range(low=0, high=10 ** 9, value=20000, mode='spinner')
    num_of_rows_after_each_distance = tr.Range(
        low=0, high=10 ** 9, value=200, mode='spinner')
