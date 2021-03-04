Derived Data Products
==========================

In addition to the data immediately available in the data products, there are a number of additional derived quantities that can be generated on the fly.
These quantities are available as part of an analysis class, and are documented below as methods of that class.

The quantities can be accessed in two ways:

1. Run the relevant method, after which the derived quantity will be available in the class's `.data` attribute. If `d` is the instance of the class, then this to get the time since the start of the simulation you would use `d.calc_time() ; d.data['time']`
2. Use the class's data retrieval methods, `get_data`, e.g. `d.get_data( 'time' )`. This also works with the more advanced `get_processed_data` and `get_selected_data`.

.. autoclass:: galaxy_dive.tests.test_utils.test_utilities.GenDerivedDataObject
    :show-inheritance:

    .. automethod:: calc_A
    .. automethod:: calc_B
