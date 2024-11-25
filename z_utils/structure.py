# pip install easy-media-utils
# python z_utils/structure.py
from tree_utils.struct_tree_out import print_tree

path = "../my_lm_log"
exclude_dirs_set = {
    "z_using_files",
    "test",
    "z_utils",
    "y_example_run",
    "z_学习案例",
    ".env",
    "x_site_building",
    "practical",
    "data",
    "pics",
    "math_beauty",
    "LICENSE",
    "README.md",
    "prompt.md",
    "requirements.txt",
    "ipynb",
    "6_代码实现",
}
print_tree(directory=path, exclude_dirs=exclude_dirs_set)
