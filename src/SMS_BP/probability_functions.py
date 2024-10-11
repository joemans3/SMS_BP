import numpy as np

# since the tophat is not a continuous probability distribution and hence non-analytic we need to define its probability function as a class


class multiple_top_hat_probability:
    """Class for the probability function of multiple top hats.
    Once initalized an object of this class can be called to return the probability at a given position.

    !!!--DO NOT CHANGE THE PARAMETERS AFTER INITALIZATION DIRECTLY. USE THE UPDATE_PARAMETERS METHOD--!!!
    """

    def __init__(
        self,
        num_subspace: int,
        subspace_centers: np.ndarray,
        subspace_radius: np.ndarray,
        density_dif: float,
        space_size: np.ndarray,
    ) -> None:
        self.num_subspace = num_subspace
        self.subspace_centers = subspace_centers
        self.subspace_radius = subspace_radius
        self.density_dif = density_dif
        self.space_size = space_size
        self.subspace_probability = self._calculate_subspace_probability(
            self.space_size, self.density_dif
        )
        self.non_subspace_probability = self._calculate_non_subspace_probability(
            self.space_size, self.density_dif, self.num_subspace, self.subspace_radius
        )

    def __call__(self, position: np.ndarray, **kwargs) -> float:
        """Returns the probability given a coordinate"""
        if not isinstance(position, np.ndarray):
            raise TypeError("Position must be a numpy array.")

        for i in range(self.num_subspace):
            # check if the position is in the subspace defined by the radius and center
            if (
                np.linalg.norm(position - self.subspace_centers[i])
                <= self.subspace_radius[i]
            ):
                return self.subspace_probability
        return self.non_subspace_probability

    def update_parameters(
        self,
        num_subspace: int | None = None,
        subspace_centers: np.ndarray | None = None,
        subspace_radius: np.ndarray | None = None,
        density_dif: float | None = None,
        space_size: np.ndarray | None = None,
    ) -> None:
        """Updates the parameters of the probability function."""
        # the None checks are not ideal but its a quick fix for now, should be updated to be *args and **kwargs checks
        if num_subspace is not None:
            self.num_subspace = num_subspace
        if subspace_centers is not None:
            self.subspace_centers = subspace_centers
        if subspace_radius is not None:
            self.subspace_radius = subspace_radius
        if density_dif is not None:
            self.density_dif = density_dif
        if space_size is not None:
            self.space_size = space_size

        self.subspace_probability = self._calculate_subspace_probability(
            self.space_size, self.density_dif
        )
        self.non_subspace_probability = self._calculate_non_subspace_probability(
            self.space_size, self.density_dif, self.num_subspace, self.subspace_radius
        )

    def _calculate_subspace_probability(
        self, space_size: np.ndarray, density_dif: float
    ) -> float:
        total_area = float(np.prod(space_size))
        return density_dif / total_area

    def _calculate_non_subspace_probability(
        self,
        space_size: np.ndarray,
        density_dif: float,
        num_subspace: int,
        subspace_radius: np.ndarray,
    ) -> float:
        total_area = float(np.prod(space_size))
        # total_subspace_area = np.sum((4.0 / 3.0) * np.pi * subspace_radius**3)
        # gamma_dif = (total_area - density_dif * total_subspace_area) / (
        #    total_area - total_subspace_area
        # )

        return 1.0 / total_area

    @property
    def num_subspace(self) -> int:
        """Returns the number of subspaces."""
        return self._num_subspace

    @num_subspace.setter
    def num_subspace(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("Number of subspaces must be an integer.")
        self._num_subspace = value

    @property
    def subspace_centers(self) -> np.ndarray:
        """Returns the centers of the subspaces."""
        return self._subspace_centers

    @subspace_centers.setter
    def subspace_centers(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Subspace centers must be a numpy array.")
        self._subspace_centers = value

    @property
    def subspace_radius(self) -> np.ndarray:
        """Returns the radius of the subspaces."""
        return self._subspace_radius

    @subspace_radius.setter
    def subspace_radius(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Subspace radius must be a numpy array.")
        self._subspace_radius = value

    @property
    def density_dif(self) -> float:
        """Returns the difference in density between the subspaces and the rest of the space."""
        return self._density_dif

    @density_dif.setter
    def density_dif(self, value: float) -> None:
        self._density_dif = value

    @property
    def space_size(self) -> np.ndarray:
        """Returns the size of the space."""
        return self._space_size

    @space_size.setter
    def space_size(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("Space size must be a numpy array.")
        self._space_size = value

    @property
    def subspace_probability(self) -> float:
        return self._subspace_probability

    @subspace_probability.setter
    def subspace_probability(self, value: float) -> None:
        self._subspace_probability = value

    @property
    def non_subspace_probability(self) -> float:
        """Returns the probability of the non-subspaces."""
        return self._non_subspace_probability

    @non_subspace_probability.setter
    def non_subspace_probability(self, value: float) -> None:
        self._non_subspace_probability = value
