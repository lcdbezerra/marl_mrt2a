DEFAULT_CONFIG = {
    "sz": (20,20),
    "n_agents": 10,
    "n_obj": [4,3,3],
    "view_range": 4,
    "comm_range": 8,
    "comm": 0,
    "render": False,
    "max_obj_lvl": 3,
    "one_hot_obj_lvl": True,
    "Lsec": 2,
    "action_grid": True,
    "share_intention": "path",
    "respawn": True,
    "discount": 0.95,
}

SZ = 20
DENSE_OBJ_DENSITY = .1
SPARSE_OBJ_DENSITY = .05

DENSE_LVL1 = {
    "n_obj": [int(DENSE_OBJ_DENSITY*SZ**2)],
    "Lsec": 1,
}
DENSE_2SECTION_LVL1 = {
    "n_obj": [int(DENSE_OBJ_DENSITY*(SZ/2)**2)], # keeps object density
    "Lsec": 2,
}
SPARSE_2SECTION_LVL1 = {
    "n_obj": [int(SPARSE_OBJ_DENSITY*(SZ/2)**2)],
    "Lsec": 2,
}
SPARSE_2SECTION_LVL12 = {
    "n_obj": 2*[int(SPARSE_OBJ_DENSITY/2*(SZ/2)**2)],
    "Lsec": 2,
}
SPARSE_2SECTION_LVL123 = {
    "n_obj": 3*[int(SPARSE_OBJ_DENSITY/3*(SZ/2)**2)],
    "Lsec": 2,
}

TEST_CONFIG = DEFAULT_CONFIG.copy()