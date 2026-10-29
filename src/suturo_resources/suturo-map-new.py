"""
The SuTuRo lab as semantic_digital_twin models it in
:class:`~semantic_digital_twin.predefined_maps.kitchen_environment.KitchenEnvironment`,
placed in the map the way :mod:`suturo_resources.suturo_map` places its own lab, with
the sofa of :class:`~suturo_resources.suturo_map.SofaLook`, and the same lab with the
kitchen of :mod:`suturo_resources.suturo_map` instead of its own.

Run it to publish the lab with the kitchen of :mod:`suturo_resources.suturo_map` to
Rviz::

    python suturo_map_new.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CounterTop,
    ShelfLayer,
    Table,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

from suturo_resources.suturo_map import (
    LabBody,
    LabLook,
    SofaLook,
    TopBoard,
    load_environment,
)

# %% the lab


@dataclass
class SuturoLab:
    """
    The SuTuRo lab of :class:`KitchenEnvironment`, which lays the lab out in a frame of
    its own, hung into the map at :attr:`map_T_room`.
    """

    map_T_room: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.33, y=0.28, yaw=0.10707963267
        )
    )
    """
    Where the frame the lab is laid out in lies in the map, the same as in
    :mod:`suturo_resources.suturo_map`.
    """

    sofa_look: SofaLook = field(default_factory=lambda: SofaLook(np.pi))
    """
    How the sofa looks, facing the coffee table in front of it.
    """

    sofa_name: str = "sofa"
    """
    The name of the sofa's body.
    """

    map_name: str = "map"
    """
    The name of the world's root body, the map.
    """

    def build(self) -> World:
        """
        :return: A world whose root is the map, holding the lab at :attr:`map_T_room`
            with the sofa of :attr:`sofa_look`.
        """
        world = World.create_with_root_body(self.map_name)
        lab = KitchenEnvironment().get_world()
        world.merge_world(
            lab,
            root_connection=FixedConnection(
                parent=world.root,
                child=lab.root,
                parent_T_connection_expression=self.map_T_room,
            ),
        )
        with world.modify_world():
            self.sofa_look.dress(world.get_body_by_name(self.sofa_name))
        return world


# %% the lab with the kitchen of suturo_map


class KitchenEnvironmentKitchenBody(StrEnum):
    """
    The bodies the kitchen of :class:`KitchenEnvironment` hangs from: the counter along
    the wall with its sink and cabinets, the oven tower, the refrigerator and the
    kitchen island.
    """

    COUNTER_TOP = "counter_top"
    DISHWASHER = "dishwasher"
    SINK_CABINET = "module_1_cabinet"
    SINK_CABINET_FACE_PLATE = "module_1_face_plate"
    OVEN_TOWER = "oven_tower"
    REFRIGERATOR = "refrigerator"
    KITCHEN_ISLAND = "kitchen_island"


@dataclass
class SuturoLabWithSuturoMapKitchen(SuturoLab):
    """
    The lab of :class:`SuturoLab` with the kitchen of :mod:`suturo_resources.suturo_map`
    in place of its own.

    Every body keeps the pose its own map gives it.
    """

    suturo_map_kitchen: tuple[LabBody, ...] = (
        LabBody.COUNTER_TOP,
        LabBody.OVEN_AREA,
        LabBody.REFRIGERATOR,
        LabBody.MIDDLE_COUNTER_TOP,
    )
    """
    The bodies of :mod:`suturo_resources.suturo_map` the kitchen is made of.
    """

    surface_color: Color = field(default_factory=Color.BEIGE)
    """
    The color of every table top, counter top and shelf layer.
    """

    top_board_thickness: float = 0.02
    """
    How thick the painted board on top of a counter of the kitchen is.
    """

    look: LabLook = field(default_factory=LabLook)
    """
    The look the kitchen of :mod:`suturo_resources.suturo_map` is drawn with.
    """

    def build(self) -> World:
        """
        :return: The lab of :meth:`SuturoLab.build` without its kitchen, holding the
            kitchen of :mod:`suturo_resources.suturo_map` where that map places it, with
            its tops painted, see :meth:`_paint_tops`.
        """
        world = super().build()
        [lab_root] = world.compute_child_kinematic_structure_entities(world.root)
        self._remove_branches(
            world,
            [
                child
                for child in world.compute_child_kinematic_structure_entities(lab_root)
                if child.name.name in set(KitchenEnvironmentKitchenBody)
            ],
        )
        kitchen = load_environment()
        self._remove_branches(
            kitchen,
            [
                child
                for child in kitchen.compute_child_kinematic_structure_entities(
                    kitchen.root
                )
                if child.name.name not in set(self.suturo_map_kitchen)
            ],
        )
        # The root of suturo_map is the map itself.
        world.merge_world(
            kitchen,
            root_connection=FixedConnection(parent=world.root, child=kitchen.root),
        )
        self._paint_tops(world)
        return world

    def _paint_tops(self, world: World) -> None:
        """
        Paint every table top, counter top and shelf layer in :attr:`surface_color`.

        A top drawn from the IAI kitchen is given a painted board of its own, which the
        rest of its look leaves room for; every other top is a board itself and is
        painted whole.

        :param world: The lab.
        """
        kitchen_looks = (
            self.look.kitchen_looks() if self.look.kitchen.is_installed else {}
        )
        top_board = TopBoard(self.top_board_thickness, self.surface_color)
        with world.modify_world():
            for annotation in world.semantic_annotations:
                if not isinstance(annotation, (Table, CounterTop, ShelfLayer)):
                    continue
                body = annotation.root
                if body.name.name in kitchen_looks:
                    replace(kitchen_looks[body.name.name], top_board=top_board).dress(
                        body
                    )
                    continue
                for shape in body.visual.shapes:
                    shape.color = self.surface_color

    @staticmethod
    def _remove_branches(world: World, roots: list[KinematicStructureEntity]) -> None:
        """
        Remove the branches hanging from ``roots``, together with every semantic
        annotation of their entities.

        :param world: The world to remove the branches from.
        :param roots: The roots of the branches.
        """
        removed = {
            entity
            for root in roots
            for entity in world.get_kinematic_structure_entities_of_branch(root)
        }
        with world.modify_world():
            for annotation in list(world.semantic_annotations):
                if removed & set(annotation.kinematic_structure_entities):
                    world.remove_semantic_annotation(annotation)
            for root in roots:
                world.remove_branch_from_world(root)


# %% showing it


def main(node_name: str = "suturo_lab_new", spin_period: float = 0.1) -> None:
    """
    Build the lab with the kitchen of :mod:`suturo_resources.suturo_map` and publish it
    to Rviz until interrupted.

    :param node_name: The name of the node the lab is published from.
    :param spin_period: How long each turn of the node waits for work, in seconds.
    """
    world = SuturoLabWithSuturoMapKitchen().build()
    rclpy.init()
    node = rclpy.create_node(node_name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    VizMarkerPublisher(_world=world, node=node)
    while rclpy.ok():
        executor.spin_once(timeout_sec=spin_period)


if __name__ == "__main__":
    main()

