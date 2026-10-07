#!/usr/bin/env python

"""
flow_accumulator_mfd_marine_deposition.py: Component to route sediment
through an existing multiple-flow-direction network.

This component reuses the MFD routing fields produced by Landlab's
existing FlowAccumulator(flow-direction=FlowDirectorMFD) and routes 
prescribed sediment flux through that network.

The component does not implement the MFD water-routing algorithm.
Instead, it uses ``flow__receiver_node``,
``flow__receiver_proportions``, and ``flow__upstream_node_order``
already present on ``FlowDirectorMFD`` component.

Through this ``FlowAccumulatorMFDSediment``component, 
sediment flux can be prescribed at each grid node and routed through the
existing MFD network.

This first implementation does not calculate erosion, deposition,
transport capacity, grain-size sorting, terrestrial-to-marine transfer,
or marine sediment transport.
"""

# Import NumPy for numerical arrays and sediment-flux calculations.
import numpy as np

# Import Landlab's base Component class so the component follows
# the standard Landlab component architecture.
from landlab import Component


# Define the sediment-routing component.
class FlowAccumulatorMFDSediment(Component):
    """Route prescribed sediment flux through an existing MFD network.

    The ``FlowAccumulator`` can use ``FlowDirectorMFD`` as its
    flow-direction method. ``FlowDirectorMFD``
    implements multiple-flow-direction routing and creates the routing
    fields:

    * ``flow__receiver_node``
    * ``flow__receiver_proportions``
    * ``flow__upstream_node_order``

    These fields are then used by ``FlowAccumulatorMFDSediment`` to route
    the prescribed sediment flux downstream through the existing MFD
    network.

    In particular, ``flow__receiver_node`` identifies the downstream
    receiver nodes, while ``flow__receiver_proportions`` gives the fraction
    of flux routed to each receiver. ``flow__upstream_node_order`` provides
    the order in which nodes are processed during downstream sediment
    routing.

    Pseudo code for the sediment-routing algorithm is as follows:

        START

            Input:
                - Landlab grid
                - Prescribed sediment flux at each node
                - MFD receiver nodes
                - MFD receiver proportions
                - Upstream node ordering

            Store the grid.

            Store the prescribed sediment flux.

            Check that one sediment-flux value exists for every grid node.

            Create:
                sediment_influx
                sediment_outflux

            Get from the grid:
                receivers
                proportions
                upstream_node_order

            Set sediment_influx
                = prescribed sediment flux

            Set sediment_outflux
                = sediment_influx

            Reverse the upstream-node ordering
                so that sediment is processed
                from upstream toward downstream.

            FOR each node in the reversed order:

                Identify the active MFD receivers.

                FOR each active receiver:

                    Read the MFD routing proportion.

                    Calculate sediment sent to that receiver:

                        branch sediment flux
                        = routing proportion
                        × sediment outflux at current node

                    Add the branch sediment flux
                    to the receiver's sediment influx.

                    Set the receiver's sediment outflux
                    equal to its accumulated sediment influx.

                END FOR

            END FOR

            Return sediment_outflux

        END


    The existing MFD flow-routing component determines where flow can go.
    This component then uses the same receiver network to determine
    where sediment goes.

    For node ``i`` and downstream receiver ``j``, the sediment flux
    routed along one MFD branch is

    Q_s(i -> j) = p_ij * Q_s(i)

    where ``Q_s`` is sediment flux and ``p_ij`` is the routing
    proportion.


    Parameters
    ----------
    grid : ModelGrid
        A Landlab model grid containing an existing MFD routing solution.

    sediment_flux : array-like
        Prescribed sediment flux supplied at every grid node.

    Notes
    -----
    This first implementation performs sediment routing only. Further
    development is needed to couple sediment routing with erosion,
    deposition, transport capacity, grain-size sorting, terrestrial-to-marine
    sediment transfer, and marine sediment transport.


    Example 
    >>> # Import the Landlab components.
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import FlowAccumulator
    >>> # Import the sediment-routing component developed for this work.
    >>> from landlab.components.flow_accum.flow_accumulator_mfd_marine_deposition 
    ...     import FlowAccumulatorMFDSediment

    >>> # Create a 10 × 10 grid with unit spacing.
    >>> mg = RasterModelGrid((10, 10), xy_spacing=(1, 1))

    >>> # Add a planar elevation field.
    >>> mg.add_field(
    ...     "topographic__elevation",
    ...     mg.node_x + mg.node_y,
    ...     at="node",
    ... )
    >>> # Get the elevation field from the grid.
    >>> elevation = mg.at_node["topographic__elevation"]

    ==Output: TOPOGRAPHIC ELEVATION ==========
    shape: (100,)
    ndim: 1
    array:
    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.  1.  2.  3.  4.  5.  6.  7.  8.
    9. 10.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.  3.  4.  5.  6.  7.  8.
    9. 10. 11. 12.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13.  5.  6.  7.  8.
    9. 10. 11. 12. 13. 14.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15.  7.  8.
    9. 10. 11. 12. 13. 14. 15. 16.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
    9. 10. 11. 12. 13. 14. 15. 16. 17. 18.]

    Elevation as 5 × 5 grid:
    [[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
    [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
    [ 2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]
    [ 3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]
    [ 4.  5.  6.  7.  8.  9. 10. 11. 12. 13.]
    [ 5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
    [ 6.  7.  8.  9. 10. 11. 12. 13. 14. 15.]
    [ 7.  8.  9. 10. 11. 12. 13. 14. 15. 16.]
    [ 8.  9. 10. 11. 12. 13. 14. 15. 16. 17.]
    [ 9. 10. 11. 12. 13. 14. 15. 16. 17. 18.]]

    >>> # Create and run the MFD flow accumulator.
    >>> fa = FlowAccumulator(
    ...     mg,
    ...     flow_director="FlowDirectorMFD",
    ... )
    >>> fa.run_one_step()

    # Retrieve the MFD receiver nodes and routing proportions.
    >>> receivers = mg.at_node["flow__receiver_node"]

    ==Output: FLOW RECEIVER NODE ==========
    shape: (100, 4)
    ndim: 2
    array:
    [[ 0 -1 -1 -1]
    [ 1 -1 -1 -1]
    [ 2 -1 -1 -1]
    [ 3 -1 -1 -1]
    [ 4 -1 -1 -1]
    [ 5 -1 -1 -1]
    [ 6 -1 -1 -1]
    [ 7 -1 -1 -1]
    [ 8 -1 -1 -1]
    [ 9 -1 -1 -1]
    [10 -1 -1 -1]
    [-1 -1 10  1]
    [-1 -1 11  2]
    [-1 -1 12  3]
    [-1 -1 13  4]
    [-1 -1 14  5]
    [-1 -1 15  6]
    [-1 -1 16  7]
    [-1 -1 17  8]
    [19 -1 -1 -1]
    [20 -1 -1 -1]
    [-1 -1 20 11]
    [-1 -1 21 12]
    [-1 -1 22 13]
    [-1 -1 23 14]
    [-1 -1 24 15]
    [-1 -1 25 16]
    [-1 -1 26 17]
    [-1 -1 27 18]
    [29 -1 -1 -1]
    [30 -1 -1 -1]
    [-1 -1 30 21]
    [-1 -1 31 22]
    [-1 -1 32 23]
    [-1 -1 33 24]
    [-1 -1 34 25]
    [-1 -1 35 26]
    [-1 -1 36 27]
    [-1 -1 37 28]
    [39 -1 -1 -1]
    [40 -1 -1 -1]
    [-1 -1 40 31]
    [-1 -1 41 32]
    [-1 -1 42 33]
    [-1 -1 43 34]
    [-1 -1 44 35]
    [-1 -1 45 36]
    [-1 -1 46 37]
    [-1 -1 47 38]
    [49 -1 -1 -1]
    [50 -1 -1 -1]
    [-1 -1 50 41]
    [-1 -1 51 42]
    [-1 -1 52 43]
    [-1 -1 53 44]
    [-1 -1 54 45]
    [-1 -1 55 46]
    [-1 -1 56 47]
    [-1 -1 57 48]
    [59 -1 -1 -1]
    [60 -1 -1 -1]
    [-1 -1 60 51]
    [-1 -1 61 52]
    [-1 -1 62 53]
    [-1 -1 63 54]
    [-1 -1 64 55]
    [-1 -1 65 56]
    [-1 -1 66 57]
    [-1 -1 67 58]
    [69 -1 -1 -1]
    [70 -1 -1 -1]
    [-1 -1 70 61]
    [-1 -1 71 62]
    [-1 -1 72 63]
    [-1 -1 73 64]
    [-1 -1 74 65]
    [-1 -1 75 66]
    [-1 -1 76 67]
    [-1 -1 77 68]
    [79 -1 -1 -1]
    [80 -1 -1 -1]
    [-1 -1 80 71]
    [-1 -1 81 72]
    [-1 -1 82 73]
    [-1 -1 83 74]
    [-1 -1 84 75]
    [-1 -1 85 76]
    [-1 -1 86 77]
    [-1 -1 87 78]
    [89 -1 -1 -1]
    [90 -1 -1 -1]
    [91 -1 -1 -1]
    [92 -1 -1 -1]
    [93 -1 -1 -1]
    [94 -1 -1 -1]
    [95 -1 -1 -1]
    [96 -1 -1 -1]
    [97 -1 -1 -1]
    [98 -1 -1 -1]
    [99 -1 -1 -1]]

    >>> proportions = mg.at_node["flow__receiver_proportions"]

    ==Output: FLOW RECEIVER PROPORTIONS ==========
    shape: (100, 4)
    ndim: 2
    array:
    [[1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [0.  0.  0.5 0.5]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]
    [1.  0.  0.  0. ]]

    >>> # Retrieve the upstream node ordering.
    >>> stack = mg.at_node["flow__upstream_node_order"]

    ==Output: FLOW UPSTREAM NODE ORDER ==========
    shape: (100,)
    ndim: 1
    array:
    [ 0  1  2  3  4  5  6  7  8  9 10 19 20 29 30 39 40 49 50 59 60 69 70 79
    80 89 90 91 92 93 94 95 96 97 98 99 11 12 21 13 22 31 14 23 32 41 15 24
    33 42 51 16 25 34 43 52 61 17 26 35 44 53 62 71 18 27 36 45 54 63 72 81
    28 37 46 55 64 73 82 38 47 56 65 74 83 48 57 66 75 84 58 67 76 85 68 77
    86 78 87 88]

    >>> # Prescribe 100 units of sediment flux at node 88.
    >>> source_node = 88
    >>> sediment_flux = np.zeros(mg.number_of_nodes)
    >>> sediment_flux[source_node] = 100.0

    >>> # Create the MFD sediment-routing component.
    >>> sediment = FlowAccumulatorMFDSediment(
    ...     mg,
    ...     sediment_flux,
    )

    >>> # Route the prescribed sediment through the MFD network.
    >>> sediment.run_one_step()

    >>> # Get the sediment influx calculated at each node.
    >>> sediment_influx = mg.at_node["sediment__influx"]

    ===Output: sediment_influx ==========
    shape: (100,)
    ndim: 1
    [  0.  10.  10.  10.   8.   6.   4.   2.   0.   0.  10.  21.  21.  19.
    16.  12.   7.   3.   1.   0.  10.  21.  23.  23.  21.  16.  11.   5.
    2.   0.  10.  19.  23.  25.  25.  22.  16.   9.   3.   0.   8.  16.
    21.  25.  27.  27.  23.  16.   6.   0.   6.  12.  16.  22.  27.  31.
    31.  25.  12.   0.   4.   7.  11.  16.  23.  31.  38.  38.  25.   0.
    2.   3.   5.   9.  16.  25.  38.  50.  50.   0.   0.   1.   2.   3.
    6.  12.  25.  50. 100.   0.   0.   0.   0.   0.   0.   0.   0.   0.
    0.   0.]

    """

    # Give the component its Landlab component name.
    _name = "FlowAccumulatorMFDSediment"

    # State that this initial implementation does not impose
    # a specific physical unit system.
    _unit_agnostic = True

    # Define the fields produced by this component.
    _info = {
        # Define the sediment influx field.
        "sediment__influx": {
            # Store sediment flux as floating-point values.
            "dtype": float,

            # This component produces the field.
            "intent": "out",

            # The field is required.
            "optional": False,

            # Use mass per unit time for the current implementation.
            "units": "mass/time",

            # Store one value per grid node.
            "mapping": "node",

            # Describe the physical meaning of the field.
            "doc": "Sediment flux arriving at each node.",
        },

        # Define the sediment outflux field.
        "sediment__outflux": {
            # Store sediment flux as floating-point values.
            "dtype": float,

            # This component produces the field.
            "intent": "out",

            # The field is required.
            "optional": False,

            # Use mass per unit time for the current implementation.
            "units": "mass/time",

            # Store one value per grid node.
            "mapping": "node",

            # Describe the physical meaning of the field.
            "doc": "Sediment flux leaving each node.",
        },
    }

    # Define the component initialization method.
    def __init__(self, grid, sediment_flux):
        """Initialize the MFD sediment-routing component.

        Parameters
        ----------
        grid : ModelGrid
            Landlab model grid containing the existing MFD routing fields.

        sediment_flux : array-like
            Prescribed sediment flux at every grid node.
        """

        # Initialize the parent Landlab Component.
        super().__init__(grid)

        # Store a reference to the model grid.
        self._grid = grid

        # Convert the prescribed sediment flux to a floating-point
        # NumPy array so that numerical operations are well defined.
        self._sediment_source = np.asarray(
            sediment_flux,
            dtype=float,
        )

        # Check that one sediment value has been supplied for every node.
        if self._sediment_source.size != grid.number_of_nodes:

            # Stop with a clear error if the input size is incorrect.
            raise ValueError(
                "sediment_flux must have one value for every grid node."
            )

        # Create the output fields declared in _info.
        self.initialize_output_fields()

        # Store a reference to the sediment influx field.
        self._sediment_influx = grid.at_node["sediment__influx"]

        # Store a reference to the sediment outflux field.
        self._sediment_outflux = grid.at_node["sediment__outflux"]

    # Define the method that performs one sediment-routing step.
    def run_one_step(self):
        """Route sediment through the existing MFD routing network."""

        # Retrieve the receiver IDs calculated by FlowDirectorMFD.
        receivers = self._grid.at_node["flow__receiver_node"]

        # Retrieve the MFD routing proportions calculated by
        # FlowDirectorMFD.
        proportions = self._grid.at_node["flow__receiver_proportions"]

        # Retrieve the downstream-to-upstream node ordering calculated
        # by FlowAccumulator's route-to-many accumulation algorithm.
        stack = self._grid.at_node["flow__upstream_node_order"]

        # Begin with the prescribed local sediment sources.
        self._sediment_influx[:] = self._sediment_source

        # With no deposition or storage in this first implementation,
        # all sediment arriving at a node is available to leave it.
        self._sediment_outflux[:] = self._sediment_influx

        # Reverse the downstream-to-upstream ordering so that sediment
        # can be propagated from upstream nodes toward downstream nodes.
        for node in stack[::-1]:

            # Identify receiver slots that represent real downstream
            # paths rather than self-receivers or unused slots.
            active = (proportions[node] > 0.0) & (
                receivers[node] != node
            )

            # Process each active MFD receiver of the current node.
            for receiver, proportion in zip(
                receivers[node, active],
                proportions[node, active],
            ):

                # Calculate the sediment flux assigned to this MFD branch.
                # Q_s(i -> j) = p_ij * Q_s(i)
                branch_flux = (
                    proportion
                    * self._sediment_outflux[node]
                )

                # Add the branch contribution to the downstream receiver.
                self._sediment_influx[receiver] += branch_flux

                # With no deposition or storage, all received sediment
                # is immediately available to leave the downstream node.
                self._sediment_outflux[receiver] = (
                    self._sediment_influx[receiver]
                )

        # Return the resulting sediment outflux field.
        return self._sediment_outflux
