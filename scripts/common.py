"""
Common python file
"""

import os


class Common:
    @staticmethod
    def change_dir(dir):
        try:
            os.chdir('%s' % dir)
        except FileNotFoundError:
            print("Directory/Folder does not exist")
