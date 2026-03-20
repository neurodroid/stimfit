import importlib
import sys


_SHELL = None


def register_shell(shell):
    global _SHELL
    _SHELL = shell
    return shell


def get_shell():
    return _SHELL


def get_shell_namespace():
    shell = get_shell()
    if shell is not None and hasattr(shell, "user_ns"):
        return shell.user_ns
    return None


def bootstrap_namespace(namespace):
    namespace.setdefault("__name__", "__console__")
    exec("from embedded_init import *", namespace, namespace)
    return namespace


def _import_or_reload_module(module_name):
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def import_module_from_path(module_path, module_name):
    namespace = get_shell_namespace()
    added_path = False

    if module_path and module_path not in sys.path:
        sys.path.append(module_path)
        added_path = True

    try:
        module = _import_or_reload_module(module_name)
        if namespace is not None:
            namespace[module_name] = module
        return module
    finally:
        if added_path:
            try:
                sys.path.remove(module_path)
            except ValueError:
                pass


def startup_banner(intro_message, loaded_message):
    return "%s%s" % (intro_message, loaded_message)
