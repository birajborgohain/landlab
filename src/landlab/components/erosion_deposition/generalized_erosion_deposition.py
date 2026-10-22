import numpy as np

from landlab import Component
from landlab import RasterModelGrid
from landlab.utils.return_array import return_array_at_node

from ..depression_finder.lake_mapper import _FLOODED

DEFAULT_MINIMUM_TIME_STEP = 0.001  # default minimum time step duration


class _GeneralizedErosionDeposition(Component):
    """Base class for erosion-deposition type components.

    More documentation here.
    """

    _name = "_GeneralizedErosionDeposition"

    _unit_agnostic = True

    _info = {
        "flow__link_to_receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "ID of link downstream of each node, which carries the discharge",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
        },
        "sediment__influx": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux (volume per unit time of sediment entering each node)",
        },
        "sediment__outflux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux (volume per unit time of sediment leaving each node)",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m**2/s",
            "mapping": "node",
            "doc": "Volumetric discharge of surface water",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
    }

    def __init__(
        self,
        grid,
        m_sp,
        n_sp,
        F_f,
        v_s,
        discharge_field="surface_water__discharge",
        dt_min=DEFAULT_MINIMUM_TIME_STEP,
        status_at_node=None,
        sea_level=0.0,
    ):
        """Initialize the GeneralizedErosionDeposition model.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        m_sp : float
            Discharge exponent (units vary)
        n_sp : float
            Slope exponent (units vary)
        F_f : float
            Fraction of eroded material that turns into "fines" that do not
            contribute to (coarse) sediment load. Defaults to zero.
        v_s : array_like of float
            Effective settling velocity for chosen grain size metric [L/T].
        discharge_field : float, field name, or array
            Discharge [L^2/T].
        dt_min : float, optional
            Only applies when adaptive solver is used. Minimum timestep that
            adaptive solver will use when subdividing unstable timesteps.
            Default values is 0.001. [T].
        """
        if F_f > 1.0 or F_f < 0.0:
            raise ValueError(f"fraction of fines must be between 0.0 and 1.0 ({F_f})")

        super().__init__(grid)

        # Boundary statuses used by this component.
        if status_at_node is None:
            # Default: reference the grid-owned array.
            self._status_at_node = grid.status_at_node
        else:
            # User-supplied component-specific array.
            status_at_node = np.asarray(status_at_node)

            if status_at_node.shape != (grid.number_of_nodes,):
                raise ValueError(
                    "status_at_node must have one value per grid node."
                )

            if not np.issubdtype(status_at_node.dtype, np.integer):
                raise TypeError("status_at_node must contain integer status codes.")

            self._status_at_node = status_at_node

        self._flow_receivers = grid.at_node["flow__receiver_node"]
        self._stack = grid.at_node["flow__upstream_node_order"]
        self._topographic__elevation = grid.at_node["topographic__elevation"]
        self._slope = grid.at_node["topographic__steepest_slope"]
        self._link_to_receiver = grid.at_node["flow__link_to_receiver_node"]
        self._cell_area_at_node = grid.cell_area_at_node
        self._sea_level = sea_level

        if isinstance(grid, RasterModelGrid):
            self._link_lengths = grid.length_of_d8
        else:
            self._link_lengths = grid.length_of_link

        self.initialize_output_fields()

        self._qs = grid.at_node["sediment__outflux"]
        self._q = return_array_at_node(grid, discharge_field)

        # For backward compatibility (remove in 3.0.0+)
        grid.at_node["sediment__flux"] = grid.at_node["sediment__outflux"]

        self._Q_to_the_m = np.zeros(grid.number_of_nodes)
        self._S_to_the_n = np.zeros(grid.number_of_nodes)
        self._depo_rate = np.zeros(grid.number_of_nodes)

        # store other constants
        self._m_sp = float(m_sp)
        self._n_sp = float(n_sp)
        self._v_s = v_s
        self._dt_min = dt_min
        self._F_f = float(F_f)

    @property
    def v_s(self):
        """Effective settling velocity for chosen grain size metric [L/T]."""
        if isinstance(self._v_s, str):
            return self.grid.at_node[self._v_s]
        else:
            return self._v_s

    @property
    def sediment_influx(self):
        """Volumetric sediment influx to each node."""
        return self.grid.at_node["sediment__influx"]

    @property
    def m_sp(self):
        """Discharge exponent (units vary)."""
        return self._m_sp

    @property
    def n_sp(self):
        """Slope exponent (units vary)."""
        return self._n_sp

    def _update_flow_link_slopes(self):
        """Updates gradient between each core node and its receiver.

        Used to update slope values between sub-time-steps, when we do not
        re-run flow routing.

        Examples
        --------
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator

        >>> rg = RasterModelGrid((3, 4))
        >>> z = rg.add_zeros("node", "topographic__elevation")
        >>> z[:] = rg.x_of_node + rg.y_of_node
        >>> fa = FlowAccumulator(rg, flow_director="FlowDirectorD8")
        >>> fa.run_one_step()
        >>> rg.at_node["topographic__steepest_slope"][5:7]
        array([1.41421356, 1.41421356])
        >>> sp = _GeneralizedErosionDeposition(rg, v_s=0.001, m_sp=0.5, n_sp=1.0, F_f=0)
        >>> z *= 0.1
        >>> sp._update_flow_link_slopes()
        >>> rg.at_node["topographic__steepest_slope"][5:7]
        array([0.14142136, 0.14142136])
        """
        self._slope[:] = (
            self._topographic__elevation
            - self._topographic__elevation[self._flow_receivers]
        ) / self._link_lengths[self._link_to_receiver]

    def _calc_hydrology(self):
        self._Q_to_the_m[:] = np.power(self._q, self._m_sp)

    def _depressions_are_handled(self):
        """Return True if a depression-handling component is present."""
        return "flood_status_code" in self._grid.at_node

    def _get_flooded_core_nodes(self):
        """Return boolean node array

        True where core node is flooded or self-draining.
        """
        # is_core = self._grid.status_at_node == self._grid.BC_NODE_IS_CORE
        # For user-supplied component-specific array
        is_core = (self._status_at_node == self._grid.BC_NODE_IS_CORE)

        if self._depressions_are_handled():
            is_flooded_core = is_core & (
                self._grid.at_node["flood_status_code"] == _FLOODED
            )
        else:
            is_flooded_core = is_core & (
                self._flow_receivers == self._grid.nodes.flatten()
            )
        return np.asarray(is_flooded_core)

    def update_status_from_elevation(self, sea_level):
        """
        Classify the component's nodes according to their elevation
        relative to sea level.

        Originally core nodes are classified as:
        - CORE when elevation >= sea_level (land nodes).
        - FIXED_VALUE when elevation < sea_level (marine nodes).

        Original external boundary nodes retain their grid status.
        """

        # Get the model grid associated with this ErosionDeposition component.
        grid = self._grid

        # Check whether the component uses the grid's shared status array.
        # If both refer to the same array, changing component statuses would
        # also change the grid statuses and affect other components.
        if self._status_at_node is grid.status_at_node:
            # Stop execution because this method requires an independent array.
            raise ValueError(
                "Supply an independent status_at_node array before "
                "using elevation-based classification."
            )

        # Create a Boolean mask identifying nodes that are currently CORE
        # according to the grid's shared boundary-status array.
        # True means the node is originally considered a core node here.
        original_core = (
            grid.status_at_node == grid.BC_NODE_IS_CORE
        )

        # Get the current topographic elevation at every grid node.
        # This is the elevation field used to distinguish land from marine nodes.
        z = self._topographic__elevation

        # Create a Boolean mask identifying nodes below sea level.
        # True means marine; False means at or above sea level.
        marine = z < sea_level

        # Copy the grid's current boundary statuses into the component's
        # independent status array. This preserves the statuses of external
        # boundary nodes before the interior nodes are reclassified.
        self._status_at_node[:] = grid.status_at_node

        # Find nodes that are both originally CORE and below sea level.
        # Assign them FIXED_VALUE status for this component so they are
        # excluded from its core-node elevation updates.
        self._status_at_node[original_core & marine] = (
            grid.BC_NODE_IS_FIXED_VALUE
        )

        # Find nodes that are both originally CORE and at or above sea level.
        # Keep them as CORE so ErosionDeposition can update their elevations.
        # The ~ operator means logical NOT, so ~marine selects non-marine nodes.
        self._status_at_node[original_core & ~marine] = (
            grid.BC_NODE_IS_CORE
        )