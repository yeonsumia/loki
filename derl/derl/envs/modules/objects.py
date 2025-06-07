import numpy as np
from lxml import etree

from derl.config import cfg
from derl.utils import mjpy as mu
from derl.utils import placement as plu
from derl.utils import sample as su
from derl.utils import xml as xu


class Objects:
    """Generate objects and add them to the env."""

    def __init__(self, random_state=None):

        self.np_random = random_state

        self.len, self.width, self.h = cfg.TERRAIN.SIZE
        self.divs = cfg.HFIELD.NUM_DIVS
        self.placement_grid = np.zeros(
            (
                self.width * cfg.HFIELD.NUM_DIVS * 2,
                self.len * cfg.HFIELD.NUM_DIVS * 2,
            )
        )

        self.obj_pos = None
        self.goal_pos = None
        # Sequence of goals for goal follower
        self.curr_goal_idx = -1
        # Sites which should be visible when rendering
        self.object_sites = []
        self.markers = []
        self.maze_layout = self.generate_maze_layout()

    def place_object(self, env, size, pos, center=None, modify_hfield=False):
        size_x, size_y, size_z = size
        os = np.asarray([size_x, size_y])
        if not pos:
            pos = plu.place_on_grid(self, os, center=center)

        if not pos:
            return None
        else:
            x, y = pos

        row_idx, col_idx = env.modules["Terrain"].pos_to_idx([x, y])
        os_in_divs = np.ceil(os * self.divs).astype(int)
        max_height = np.max(
            env.metadata["hfield"][
                row_idx - os_in_divs[1] : row_idx + os_in_divs[1],
                col_idx - os_in_divs[0] : col_idx + os_in_divs[0],
            ]
        )
        pos = [x, y, size_z + max_height + 0.1]

        if modify_hfield:
            env.metadata["hfield"][
                row_idx - os_in_divs[1] : row_idx + os_in_divs[1],
                col_idx - os_in_divs[0] : col_idx + os_in_divs[0],
            ] = size_z + max_height

        return pos

    def add_box(self, env, idx, material="self"):
        bs = cfg.OBJECT.BOX_SIDE
        pos = self.place_object(
            env, [bs] * 3, cfg.OBJECT.BOX_POS, center=self.pos_to_grid_idx([0, 0])
        )
        self.obj_pos = pos
        name = "box/{}".format(idx)
        box = xu.body_elem(name, pos)
        # Add joint for box
        box.append(xu.joint_elem("{}/root".format(name), "free", "free"))
        # Add box geom
        box.append(
            etree.Element(
                "geom",
                {
                    "name": name,
                    "type": "box",
                    "size": xu.arr2str([bs, bs, bs]),
                    "condim": "3",
                    "mass": str(cfg.OBJECT.BOX_MASS),
                    "material": material,
                },
            )
        )
        # Add site at each face center
        face_sites = [
            [bs, 0, 0],
            [-bs, 0, 0],
            [0, bs, 0],
            [0, -bs, 0],
            [0, 0, bs],
            [0, 0, -bs],
        ]
        for i, fs in enumerate(face_sites):
            site_name = "box/face/{}/{}".format(i, idx)
            box.append(xu.site_elem(site_name, fs, "box_face_site"))
            self.object_sites.append(site_name)
        return box

    def add_ball(self, env, idx):
        bs = cfg.OBJECT.BALL_RADIUS
        pos = self.place_object(
            env, [bs] * 3, None, center=self.pos_to_grid_idx([0, 0])
        )
        self.obj_pos = pos
        name = "ball/{}".format(idx)
        ball = xu.body_elem(name, pos)
        # Add joint for ball
        ball.append(xu.joint_elem("{}/root".format(name), "free", "free"))
        # Add ball geom
        # Solref values taken from dm_control, explation:
        # http://www.mujoco.org/book/modeling.html (restitution section)
        ball.append(
            etree.Element(
                "geom",
                {
                    "name": name,
                    "type": "sphere",
                    "size": str(bs),
                    "condim": "6",
                    "density": "600.0",
                    "material": "ball",
                    "priority": "1",
                    "friction": "0.7 0.005 0.005",
                    "solref": "-10000 -30"
                },
            )
        )
        return ball

    def add_goal(self, env):
        gs = 0.5
        gh = 0.01
        center = None
        if self.obj_pos is not None:
            center = self.pos_to_grid_idx(self.obj_pos[:2])
        self.goal_pos = self.place_object(
            env,
            [gs, gs, gh],
            cfg.OBJECT.GOAL_POS,
            center=center
        )
        self.markers.append(
            {
                "label": "",
                "size": np.array([gs, gs, gh]),
                "rgba": np.array([1, 0, 0, 0.4]),
                "pos": np.array(self.goal_pos),
            }
        )

    def add_obstacles(self, env):
        obstacles = []
        for idx in range(cfg.OBJECT.NUM_OBSTACLES):
            bl = su.sample_from_range(
                cfg.OBJECT.OBSTACLE_LEN_RANGE, self.np_random
            )
            bw = su.sample_from_range(
                cfg.OBJECT.OBSTACLE_WIDTH_RANGE, self.np_random
            )
            obstacle_dims = [bl, bw, 2]
            pos = self.place_object(
                env, obstacle_dims, None, modify_hfield=True
            )
            if not pos:
                continue
            name = "obstacle/{}".format(idx)
            box = xu.body_elem(name, pos)
            # Add box geom
            box.append(
                etree.Element(
                    "geom",
                    {
                        "name": name,
                        "type": "box",
                        "size": xu.arr2str(obstacle_dims),
                        "condim": "3",
                        "mass": "1.0",
                        "material": "self",
                    },
                )
            )
            obstacles.append(box)
        return obstacles
    
    def add_bumps(self, env):
        obstacles = []
        for idx in range(cfg.OBJECT.NUM_BUMPS):
            bl = su.sample_from_range(
                cfg.OBJECT.BUMP_LEN_RANGE, self.np_random
            )
            bw = su.sample_from_range(
                cfg.OBJECT.BUMP_WIDTH_RANGE, self.np_random
            )
            bh = su.sample_from_range(
                cfg.OBJECT.BUMP_HEIGHT_RANGE, self.np_random
            )
            obstacle_dims = [bl, bw, bh]
            pos = self.place_object(
                env, obstacle_dims, None, modify_hfield=True
            )
            if not pos:
                continue
            name = "bump/{}".format(idx)
            box = xu.body_elem(name, pos)
            # Add box geom
            box.append(
                etree.Element(
                    "geom",
                    {
                        "name": name,
                        "type": "box",
                        "size": xu.arr2str(obstacle_dims),
                        "condim": "3",
                        "mass": "1.0",
                        "material": "self",
                    },
                )
            )
            obstacles.append(box)
        return obstacles


    def generate_maze_layout(self):
        """Generate a simple maze layout as a 2D grid where 1 is a wall and 0 is a path."""
        # Example hardcoded layout, 1 is wall, 0 is path
        # open txt file to get 2d list
        # return np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                  [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                  [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                  [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        #                  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #                  [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        #                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        #                  [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
        #                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        #                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        #                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        #                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
        #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
        # return np.array([[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        #                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        #                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        #                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        #                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        #                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        #                  [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        #                  [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        #                  [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        #                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        #                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        #                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        #                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        if cfg.ENV.TASK == "maze_nav":
            # return np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
            #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
            return np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]])
        elif cfg.ENV.TASK == "maze_exploration":
            return np.array([[1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                            [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
                            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                            [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
                            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]])
        else:
            return None
            # raise ValueError("Invalid task for maze layout generation")

    def add_maze_walls(self, env):
        """Add walls to the environment based on the maze layout."""
        wall_height = cfg.OBJECT.WALL_HEIGHT
        wall_thickness = cfg.OBJECT.WALL_THICKNESS
        walls = []

        for row in range(self.maze_layout.shape[0]):
            for col in range(self.maze_layout.shape[1]):
                if self.maze_layout[row, col] == 1:  # Wall
                    wall_dims = [wall_thickness, wall_thickness, wall_height]
                    # pos = self.maze_grid_idx_to_pos(np.asarray([row, col])) + [wall_height / 2]
                    # modify hfield to account for wall - agents are expected to avoid walls
                    pos = self.place_object(
                        env, wall_dims, None, idx=self.maze_grid_idx_to_grid(np.asarray([row, col])), modify_hfield=True
                    )
                    name = f"wall/{row}_{col}"
                    wall = xu.body_elem(name, pos)
                    wall.append(
                        etree.Element(
                            "geom",
                            {
                                "name": name,
                                "type": "box",
                                "size": xu.arr2str(wall_dims),
                                "condim": "3",
                                "mass": "1.0",
                                "material": "self",
                            },
                        )
                    )
                    walls.append(wall)
        return walls

    def add_goal_at_final_pos(self, env):
        """Add a goal object at the final position in the maze (bottom-right corner)."""
        goal_size = 1.0
        goal_height = 0.01
        final_pos = [self.maze_layout.shape[0] - 1, self.maze_layout.shape[1] - 2]

        # Convert final grid position to Mujoco world position
        # self.goal_pos = self.maze_grid_idx_to_pos(np.asarray(final_pos)) + [goal_height + 1]
        self.goal_pos = self.place_object(
                        env, [goal_size, goal_size, goal_height], None, idx=self.maze_grid_idx_to_grid(np.asarray(final_pos)), modify_hfield=True
                    )
        # Create goal marker
        self.markers.append(
            {
                "label": "",
                "size": np.array([goal_size, goal_size, goal_height]),
                "rgba": np.array([1, 0, 0, 0.4]),
                "pos": np.array(self.goal_pos),
            }
        )
    
    def add_bumps(self, env):
        obstacles = []
        for idx in range(cfg.OBJECT.NUM_BUMPS):
            bl = su.sample_from_range(
                cfg.OBJECT.BUMP_LEN_RANGE, self.np_random
            )
            bw = su.sample_from_range(
                cfg.OBJECT.BUMP_WIDTH_RANGE, self.np_random
            )
            bh = su.sample_from_range(
                cfg.OBJECT.BUMP_HEIGHT_RANGE, self.np_random
            )
            obstacle_dims = [bl, bw, bh]
            pos = self.place_object(
                env, obstacle_dims, None, modify_hfield=True
            )
            if not pos:
                continue
            name = "bump/{}".format(idx)
            box = xu.body_elem(name, pos)
            # Add box geom
            box.append(
                etree.Element(
                    "geom",
                    {
                        "name": name,
                        "type": "box",
                        "size": xu.arr2str(obstacle_dims),
                        "condim": "3",
                        "mass": "1.0",
                        "material": "self",
                    },
                )
            )
            obstacles.append(box)
        return obstacles

    def modify_xml_step(self, env, root, tree):
        worldbody = root.findall("./worldbody")[0]
        xml_elems = []

        # Add maze walls if the task is maze_navigation
        if cfg.ENV.TASK == "maze_nav" or cfg.ENV.TASK == "maze_exploration":
            xml_elems.extend(self.add_maze_walls(env))
            self.add_goal_at_final_pos(env)  # Add goal at maze exit
        
        # Add box if task is manipulation
        if cfg.ENV.TASK in ["manipulation", "push_box_incline"]:
            if cfg.OBJECT.TYPE == "box":
                xml_elems.append(self.add_box(env, 1))
            else:
                xml_elems.append(self.add_ball(env, 1))

        if cfg.ENV.TASK in ["manipulation", "point_nav", "push_box_incline"]:
            self.add_goal(env)

        if cfg.ENV.TASK == "obstacle":
            xml_elems.extend(self.add_obstacles(env))
        
        if cfg.ENV.TASK == "bump":
            xml_elems.extend(self.add_bumps(env))

        for elem in xml_elems:
            worldbody.append(elem)

        env.metadata["markers"] = self.markers
        env.metadata["object_sites"] = self.object_sites

        # xu.save_etree_as_xml(tree, "1.xml")
        # np.save("outfile.npy", env.metadata["hfield"])

    def modify_sim_step(self, env, sim):
        self.obj_qpos_idxs = np.array(
            mu.qpos_idxs_from_joint_prefix(sim, cfg.OBJECT.TYPE)
        )
        self.obj_qvel_idxs = np.array(
            mu.qvel_idxs_from_joint_prefix(sim, cfg.OBJECT.TYPE)
        )

    def observation_step(self, env, sim):
        if cfg.ENV.TASK in ["manipulation", "push_box_incline"]:
            return self.manipulation_obs_step(env, sim)
        elif cfg.ENV.TASK in ["point_nav"]:
            return self.point_nav_obs_step(env, sim)
        elif cfg.ENV.TASK in ["exploration"]:
            return self.exploration_obs_step(env, sim)
        elif cfg.ENV.TASK == "maze_nav":
            return self.maze_nav_obs_step(env, sim)
        elif cfg.ENV.TASK == "maze_exploration":
            return self.maze_exploration_obs_step(env, sim)
        elif cfg.ENV.TASK == "obstacle":
            return {}
        elif cfg.ENV.TASK == "bump":
            return {}
        else:
            raise ValueError("Task not supported: {}".format(cfg.ENV.TASK))

    def manipulation_obs_step(self, env, sim):
        pos = sim.data.qpos.flat.copy()
        vel = sim.data.qvel.flat.copy()

        obj_pos = pos[self.obj_qpos_idxs][:3]
        obj_vel = vel[self.obj_qvel_idxs][:3]

        obj_rot_vel = vel[self.obj_qvel_idxs][3:]

        # Convert obj pos, vel and rot_vel in torso frame
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")
        agent_qvel_idxs = env.modules["Agent"].agent_qvel_idxs
        agent_vel = vel[agent_qvel_idxs][:3]

        obj_rel_pos = obj_pos - torso_pos
        obj_rel_vel = obj_vel - agent_vel

        goal_rel_pos = self.goal_pos - torso_pos
        obj_state = np.vstack(
            (obj_rel_pos, obj_rel_vel, obj_rot_vel, goal_rel_pos)
        )
        obj_state = obj_state.dot(torso_frame).ravel()
        return {"obj": obj_state}

    def point_nav_obs_step(self, env, sim):
        # Convert box pos, vel and rot_vel in torso frame
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")

        goal_rel_pos = self.goal_pos - torso_pos
        goal_state = goal_rel_pos.dot(torso_frame).ravel()
        return {"goal": goal_state}

    def maze_nav_obs_step(self, env, sim):
        # Convert box pos, vel and rot_vel in torso frame
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")

        goal_rel_pos = self.goal_pos - torso_pos
        goal_state = goal_rel_pos.dot(torso_frame).ravel()
        return {"goal": goal_state}

    def exploration_obs_step(self, env, sim):
        x_pos, y_pos, _ = sim.data.get_body_xpos("torso/0")
        row_idx, col_idx = self.pos_to_grid_idx([x_pos, y_pos])
        return {"placement_idx": np.asarray([row_idx, col_idx])}
    
    def maze_exploration_obs_step(self, env, sim):
        # Convert box pos, vel and rot_vel in torso frame
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        torso_pos = sim.data.get_body_xpos("torso/0")
        x_pos, y_pos, _ = torso_pos

        goal_rel_pos = self.goal_pos - torso_pos
        goal_state = goal_rel_pos.dot(torso_frame).ravel()

        row_idx, col_idx = self.pos_to_grid_idx([x_pos, y_pos])
        return {"placement_idx": np.asarray([row_idx, col_idx]), "goal": goal_state}

    def maze_navigation_obs_step(self, env, sim):
        """Provide observation data specific to maze navigation task."""
        torso_frame = sim.data.get_body_xmat("torso/0").reshape(3, 3)
        # Get agent's current position in MuJoCo world coordinates
        x_pos, y_pos, _ = sim.data.get_body_xpos("torso/0")
        
        # Convert agent position to grid coordinates
        agent_grid_idx = self.pos_to_grid_idx([x_pos, y_pos])
        
        # Get the fixed goal position (already in grid coordinates)
        goal_grid_idx = self.pos_to_grid_idx(self.goal_pos[:2])
        
        # Calculate relative position to the goal
        relative_position_to_goal = np.array(goal_grid_idx) - np.array(agent_grid_idx)
        
        # Prepare observation dictionary
        observation = {
            "goal": relative_position_to_goal.dot(torso_frame).ravel()
        }
        
        return observation

    def grid_idx_to_pos(self, idx):
        """Convert from grid --> mujoco pos."""
        idx = idx / self.divs
        row_idx, col_idx = idx
        x_pos = col_idx - cfg.TERRAIN.START_FLAT
        y_pos = row_idx - self.width
        pos = [x_pos, y_pos]
        pos = [round(_, 2) for _ in pos]
        return pos
    
    def maze_grid_idx_to_pos(self, idx):
        """Convert from grid --> mujoco pos."""
        idx = idx * 4
        row_idx, col_idx = idx
        x_pos = col_idx - cfg.TERRAIN.START_FLAT + 3
        y_pos = row_idx - self.width + 2
        pos = [x_pos, y_pos]
        pos = [round(_, 2) for _ in pos]
        return pos
    
    def maze_grid_idx_to_grid(self, idx):
        """Convert from grid --> mujoco pos."""
        offset = int(self.divs * cfg.OBJECT.WALL_THICKNESS)
        idx = idx * 2 * cfg.OBJECT.WALL_THICKNESS * self.divs
        row_idx, col_idx = idx
        row_idx = int(row_idx)
        col_idx = int(col_idx)
        return [row_idx + offset, col_idx + offset]

    def pos_to_grid_idx(self, pos):
        """Convert from mujoco pos to grid."""
        x_pos, y_pos = pos
        row_idx = y_pos + self.width
        col_idx = x_pos + cfg.TERRAIN.START_FLAT
        idx = [row_idx * self.divs, col_idx * self.divs]
        idx = [int(_) for _ in idx]
        return idx

    def assert_cfg(self):
        pass
