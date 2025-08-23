#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 14:14:55 2025

@author: yogehs
"""

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from .parsejpkh5header import get_attributes_matching,properties_section
import numpy as np
from typing import Any, Iterator, NoReturn, cast, Union, Dict
from abc import ABC, abstractmethod
import h5py
# from jpk_h5.parsejpkh5header import properties_section, get_attributes_matching


class Grid:
    """A regular grid of size i_length × j_length, inclined by an angle ϑ."""

    def __init__(
        self,
        center: tuple[float, float],
        duv: tuple[float, float],
        theta: float,
        reflect: bool,
        ij_lengths: tuple[int, int],
    ) -> None:
        (self.x_center, self.y_center) = center
        (self.du, self.dv) = duv
        self.theta = theta
        if reflect:
            raise Exception("Not implemented: Grid(…, reflect=True)")
        self.reflect = reflect
        (self.i_length, self.j_length) = ij_lengths

        cos = np.cos(theta)
        sin = np.sin(theta)
        self.dux, self.dvx = self.du * cos, -self.dv * sin
        self.duy, self.dvy = self.du * sin, self.dv * cos

    @staticmethod
    def from_properties(props: Union[dict[str, str], h5py.AttributeManager]) -> "Grid":
        # Pixel counts
        i_length = int(props["ilength"])  # fast direction
        j_length = int(props["jlength"])  # slow direction

        # Geometric length
        u_length = float(props["ulength"])  # fast direction
        v_length = float(props["vlength"])  # slow direction

        return Grid(
            center=(float(props["xcenter"]), float(props["ycenter"])),
            duv=(u_length / i_length, v_length / j_length),
            theta=float(props["theta"]),
            reflect=props["reflect"] != "false",
            ij_lengths=(i_length, j_length),
        )

    def get_position(self, i: int, j: int) -> tuple[float, float]:
        i1 = i - self.i_length / 2
        j1 = j - self.j_length / 2
        return (
            self.x_center + i1 * self.dux + j1 * self.dvx,
            self.y_center + i1 * self.duy + j1 * self.dvy,
        )

    def __repr__(self) -> str:
        return (
            "Grid("
            f"center=({self.x_center}, {self.y_center}),"
            f" duv=({self.du}, {self.dv}),"
            f" theta={self.theta},"
            f" reflect={self.reflect},"
            f" ij_lengths=({self.i_length}, {self.j_length})"
            ")"
        )


class Numbering(ABC):
    """A scheme for traversing grid points.

    The indices i and j index the fast and slow direction, respectively.
    """

    name = "UNDEFINED_IN_ABSTACT_BASE_CLASS"

    @staticmethod
    def from_name(name: str, grid: Grid) -> "Numbering":
        for cls in [LeftToRight, BackAndForth, RightToLeft]:
            if cls.name == name:
                return cls(grid)  # type: ignore
        raise Exception(f"Unknown Numbering type '{name}'")

    @abstractmethod
    def get_ij_indices(self, idx: int) -> tuple[int, int]:
        """Return the i and j indices for the idx-th point of the traversal."""
        raise Exception("Not implemented in abstract base class")

    def get_i_index(self, idx: int) -> int:
        """Return the i (=fast) index for the idx-th point of the traversal."""
        i_index, _ = self.get_ij_indices(idx)
        return i_index

    def get_j_index(self, idx: int) -> int:
        """Return the j (=slow) index for the idx-th point of the traversal."""
        _, j_index = self.get_ij_indices(idx)
        return j_index


class LeftToRight(Numbering):
    """Left-to-right numbering.

    E.g.
       16 → 17 → 18 → 19 → 20
        ⮤← ← ← ← ← ← ← ← ← ↰
       11 → 12 → 13 → 14 → 15
        ⮤← ← ← ← ← ← ← ← ← ↰
        6 →  7 →  8 →  9 → 10
        ⮤← ← ← ← ← ← ← ← ← ↰
        1 →  2 →  3 →  4 →  5

    """

    name = "left-to-right"

    def __init__(self, grid: Grid):
        self.grid = grid

    def get_ij_indices(self, idx: int) -> tuple[int, int]:
        """Return the i and j indices for the idx-th point of the traversal."""
        i_index = idx % self.grid.i_length
        j_index = idx // self.grid.i_length
        return i_index, j_index


class BackAndForth(Numbering):
    """Back-and-forth (boustrophedonic) numbering.

    E.g.
       20 ← 19 ← 18 ← 17 ← 16
                            ↑
       11 → 12 → 13 → 14 → 15
        ↑
       10 ←  9 ←  8 ←  7 ←  6
                            ↑
        1 →  2 →  3 →  4 →  5

    """

    name = "back-and-forth"

    def __init__(self, grid: Grid):
        self.grid = grid

    def get_ij_indices(self, idx: int) -> tuple[int, int]:
        """Return the i and j indices for the idx-th point of the traversal."""
        j_index = idx // self.grid.i_length

        i1 = idx % self.grid.i_length
        if j_index % 2 == 0:
            i_index = i1
        else:
            i_index = self.grid.i_length - 1 - i1

        return i_index, j_index


class RightToLeft(Numbering):
    """Right-to-left numbering.

    E.g.
       20 ← 19 ← 18 ← 17 ← 16
        ⮣ → → → → → → → → →⮥
       15 ← 14 ← 13 ← 12 ← 11
        ⮣ → → → → → → → → →⮥
       10 ←  9 ←  8 ←  7 ←  6
        ⮣ → → → → → → → → →⮥
        5 ←  4 ←  3 ←  2 ←  1

    """

    def __init__(self, grid: Grid):
        self.grid = grid

    name = "right-to-left"

    def get_ij_indices(self, idx: int) -> tuple[int, int]:
        """Return the i and j indices for the idx-th point of the traversal."""
        i_index = self.grid.i_length - 1 - idx % self.grid.i_length
        j_index = idx // self.grid.i_length
        return i_index, j_index


class GridPositionPattern:
    """A Grid, together with an order in which we traverse it."""

    def __init__(self, grid: Grid, numbering: Numbering) -> None:
        self.grid = grid
        self.numbering = numbering

    @staticmethod
    def from_properties(
        props: Union[dict[str, str], h5py.AttributeManager]
    ) -> "GridPositionPattern":

        grid_section = properties_section(props, "grid")
        grid = Grid.from_properties(grid_section)
        numbering = Numbering.from_name(props["numbering"], grid)
        return GridPositionPattern(grid, numbering)

    def get_position(self, idx: int) -> tuple[float, float]:
        i, j = self.numbering.get_ij_indices(idx)
        return self.grid.get_position(i, j)

    def get_idx(self, idx: int) -> tuple[float, float]:
        i, j = self.numbering.get_ij_indices(idx)
        return [i, j]


def get_valid_position(grid_position_pattern, valid_indices, valid_idx: int) -> tuple[float, float]:

    try:
        grid_idx = valid_indices[valid_idx]
        print(grid_idx)
    except IndexError:
        raise Exception(
            f"Cannot select {valid_idx}-th valid position"
            f" (there are only {len(valid_indices)} of them)"
        )
    return grid_position_pattern.get_position(grid_idx)


# %%
# prop = get_attributes_matching(
#     "multi-scan-series.map.header.position-pattern", top.attrs)

# grid_section = properties_section(prop, "grid")

# file_path = '/Users/yogehs/Documents/ATIP_PhD/CTC_mech_data/AFM/20250811_CTC45/sample1/smart-rectangle5-2025.08.11-15.51.01.680.h5-jpk'

# uff = loadfile(file_path)
# file_metadata = uff.filemetadata
# h5file = h5py.File(file_path, "r")
# top_group = file_metadata['top_group']
# top = h5file[top_group]
# positions = top["Position_Values"]

# grid_position_pattern = GridPositionPattern.from_properties(
#     get_attributes_matching(
#         "multi-scan-series.map.header.position-pattern", top.attrs))

# valid_indices_ds = top["meta-data"]["valid-indices"]
# valid_indices = np.asarray(top["meta-data"]["valid-indices"]).flatten()


# # %%
# valid_idx = 0
# grid_idx = valid_indices[valid_idx]
# print(grid_position_pattern.get_position(grid_idx), 'position')
# print(grid_position_pattern.get_idx(grid_idx), 'index')

# # get_valid_position(grid_position_pattern, valid_indices, 4000)

# # %%position array
# coords = np.full((file_metadata["num_y_pixels"],
#                  file_metadata["num_x_pixels"]), np.nan)

# for valid_idx in range(file_metadata["Entry_tot_nb_curve"]):
#     grid_idx = file_metadata['valid_indices'][valid_idx]
#     [i, j] = grid_position_pattern.get_idx(grid_idx)
#     # print(grid_position_pattern.get_idx(grid_idx), 'index')
#     coords[j][i] = valid_idx
#     # filemetadata["num_y_pixels"], UFF.filemetadata["num_x_pixels"]


# fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
# sns.heatmap(uff.imagedata['Slope'], vmin=0.000, vmax=0.004,
#             annot=coords, fmt="", annot_kws={"size": 2}, ax=ax)
# plt.title('Slope')
# ax.invert_yaxis()
# # plt.show()

# plt.savefig("heatmap.svg", format="svg", bbox_inches="tight", dpi=300)
