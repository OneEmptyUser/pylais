import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # THIS IS TO REMOVE A WARNING
import sys
# Añadir el directorio raíz del proyecto a sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../my_pkg')))
import tensorflow as tf

class TestUtils:
    
    def test_flatTensor3D(self):
        return True