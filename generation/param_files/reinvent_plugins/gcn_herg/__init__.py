from .herg import GCNHergComponent

def register_plugin():
    return [("herg", GCNHergComponent)]
